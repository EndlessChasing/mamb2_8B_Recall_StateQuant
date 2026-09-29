"""TRAIN-only S16 readout-damage calibration; original runtime files stay intact.

The S16 arithmetic is copied from state_codec._s16_scan (see that file and
THIRD_PARTY_NOTICES.md for attribution). Extra statistics never feed recurrence.
Scores approximate single-step loss from erasing a carried coordinate, with
fixed inputs and no coordinate cross-covariances. They do not estimate total
sequence loss or guarantee that a tier permutation improves PPL/recall.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
from . import state_codec as codec
from .state_quant import StateQuant


@triton.jit
def _readout_collect_scan(X, DT, A, B, C, D, DB, S, Y, Abs, Score,
                          L:tl.constexpr,H:tl.constexpr,P:tl.constexpr,G:tl.constexpr,TILE:tl.constexpr):
    b,h,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    p=tile*TILE+tl.arange(0,TILE)
    n=tl.arange(0,128)
    mask=p<P
    offsets=((b*H+h)*P+p[:,None])*128+n[None,:]
    state=tl.load(S+offsets,mask=mask[:,None],other=0).to(tl.float32)
    av=tl.load(A+h).to(tl.float32)
    dv=tl.load(D+h).to(tl.float32)
    bias=tl.load(DB+h).to(tl.float32)
    group=h//(H//G)
    abs_sum=tl.full((TILE,128),0.,tl.float32)
    score_sum=tl.full((TILE,128),0.,tl.float32)
    for t in range(L):
        xoff=((b*L+t)*H+h)*P+p
        x=tl.load(X+xoff,mask=mask,other=0).to(tl.float32)
        dt=tl.load(DT+(b*L+t)*H+h).to(tl.float32)+bias
        dt=tl.where(dt<=20.,tl.math.log(tl.math.exp(dt)+1.),dt)
        decay=tl.exp(av*dt)
        boff=((b*L+t)*G+group)*128+n
        bv=tl.load(B+boff).to(tl.float32)
        cv=tl.load(C+boff).to(tl.float32)
        # This is the previous rounded FP16 carry, before this token's write.
        # Multiply C first so statistics do not introduce a reusable state*decay
        # node that can change fusion of the original recurrent update.
        carry_readout=(state*cv[None,:])*decay
        score_sum+=tl.where(mask[:,None],carry_readout*carry_readout,0.)
        dB=bv*dt
        state=state*decay+dB[None,:]*x[:,None]
        out=tl.sum(state*cv[None,:],1)
        out+=x*dv
        tl.store(Y+xoff,out,mask=mask)
        state=state.to(tl.float16).to(tl.float32)
        abs_sum+=tl.where(mask[:,None],tl.abs(state),0.)
    tl.store(S+offsets,state,mask=mask[:,None])
    ntile=tl.cdiv(P,TILE)
    off=((b*H+h)*ntile+tile)*128+n
    tl.store(Abs+off,tl.sum(abs_sum,0))
    tl.store(Score+off,tl.sum(score_sum,0))


def _public_tiles(value,p):
    b,h,tiles,n=value.shape
    target=triton.cdiv(p,codec.RP)
    need=target*(codec.RP//codec.S16_TILE)
    if tiles!=need:
        value=torch.cat((value,torch.zeros((b,h,need-tiles,n),device=value.device,dtype=value.dtype)),2)
    return value.reshape(b,h,target,codec.RP//codec.S16_TILE,n).sum(3)


@torch.no_grad()
def readout_collect_scan(x,dt,A,B,C,D,dt_bias,state):
    """Return exact S16 readouts, abs-carry sums, squared carry-readout sums.

    Both sum tensors use [batch,head,ceil(P/16),128] FP32. State is updated
    in place exactly as the original S16 scan; only offline sums are added.
    """
    if x.ndim!=4 or x.dtype!=torch.float16 or not x.is_cuda:
        raise ValueError('Require CUDA FP16 x[batch,time,heads,P]')
    b,length,h,p=x.shape
    if length<1 or state.mode!='s16' or state.shape!=(b,h,p,128):
        raise ValueError('Require positive length and matching S16 state')
    codec._validate_state(state,x.device)
    if B.ndim!=4:
        raise ValueError('B must have four axes')
    g=B.shape[2]
    if g<1 or h%g:
        raise ValueError('Heads must be divisible by groups')
    for value,shape in ((dt,(b,length,h)),(A,(h,)),(B,(b,length,g,128)),
                        (C,(b,length,g,128)),(D,(h,)),(dt_bias,(h,))):
        if value.device!=x.device or tuple(value.shape)!=shape:
            raise ValueError('Input geometry/device differs')
    if A.dtype!=torch.float32 or any(t.dtype!=torch.float16 for t in (dt,B,C)):
        raise ValueError('A must be FP32; dt/B/C must be FP16')
    if any(t.dtype not in (torch.float16,torch.float32) for t in (D,dt_bias)):
        raise ValueError('D/dt_bias require FP16 or FP32')
    x,dt,A,B,C,D,dt_bias=(v.contiguous() for v in (x,dt,A,B,C,D,dt_bias))
    out=torch.empty_like(x)
    shape=(b,h,triton.cdiv(p,codec.S16_TILE),128)
    abs_sum=torch.empty(shape,device=x.device,dtype=torch.float32)
    score=torch.empty_like(abs_sum)
    with torch.cuda.device(x.device):
        _readout_collect_scan[(b,h,shape[2])](x,dt,A,B,C,D,dt_bias,state.tensors['state'],out,abs_sum,score,
            length,h,p,g,codec.S16_TILE,num_warps=4)
    return out,_public_tiles(abs_sum,p),_public_tiles(score,p)


def equalizer_exponents(mean_abs):
    """Round ties-to-even log2 range ratios, clamp[-8,8], in original coordinates."""
    if (mean_abs.dtype!=torch.float64 or tuple(mean_abs.shape)!=(56,8,128)
            or not bool(torch.isfinite(mean_abs).all()) or bool((mean_abs<0).any())):
        raise ValueError('Require finite nonnegative FP64 means[56,8,128]')
    values=mean_abs.clamp_min(1e-12)
    # log2(value/geomean(value)) evaluated in log space to avoid intermediate
    # overflow; geomean is per layer/group across all128 original coordinates.
    log2=values.log2()
    return (log2-log2.mean(-1,keepdim=True)).round().clamp(-8,8).to(torch.int8).contiguous()


class ReadoutCalibration(StateQuant):
    """Own controller override only; no module-global scan replacement."""
    def __init__(self,model):
        super().__init__(model,'s16',collect_stats=True)
        self._readout_sums=[None]*len(self._mixers)

    def _accumulate_readout(self,index,score,batch,length):
        if (score.dtype!=torch.float32 or score.device!=self.device
                or tuple(score.shape)!=(batch,128,4,128)):
            raise RuntimeError('Readout score workspace geometry differs')
        total=score.reshape(batch,8,16,4,128).double().sum(dim=(0,2,3))
        if self._readout_sums[index] is None:
            self._readout_sums[index]=total
        else:
            self._readout_sums[index].add_(total)

    def reset_statistics(self):
        super().reset_statistics()
        self._readout_sums=[None]*len(self._mixers)

    def statistics(self):
        result=super().statistics()
        if any(s is None for s in self._readout_sums):
            raise RuntimeError('All56 layers must collect readout damage scores')
        sums=torch.stack(self._readout_sums).detach().cpu()
        result.update(sum_readout_score=sums,
            mean_readout_score=sums/result['sample_count_per_group'][:,None,None],
            readout_score_semantics='mean((exp(A*softplus(dt+bias))*previous_FP16_carry*C_current)^2); averaged over batch,time,16heads/group,64P; cross-coordinate covariance ignored')
        return result

    def cache_breakdown(self):
        result=super().cache_breakdown()
        result['calibration_workspace_bytes']+=sum(v.numel()*v.element_size() for v in self._readout_sums if v is not None)
        batch=self._batch_size or 0
        result['calibration_native_stat_buffer_bytes']=2*batch*128*16*128*4
        result['calibration_public_stat_buffer_bytes']=2*batch*128*4*128*4
        # These count only score/abs buffers, excluding activations, scan registers and allocator reserve.
        return result

    @torch.no_grad()
    def _forward(self, index, mx, u, seqlen, seq_idx, cu_seqlens, inference_params):
        if not self._installed or self._failed:
            raise RuntimeError("Controller is inactive or failed; reset before reusing it")
        if any(value is not None for value in (seqlen, seq_idx, cu_seqlens, inference_params)):
            raise ValueError("Use controller-owned caches; native InferenceParams/varlen are unsupported")
        if (not isinstance(u, torch.Tensor) or u.ndim != 3 or u.dtype != torch.float16
                or u.device != self.device or u.shape[-1] != 4096 or u.shape[1] == 0):
            raise ValueError("Mixer input must be CUDA FP16 [batch,tokens,4096]")
        if not self._cache:
            if index != 0:
                raise RuntimeError("A new request must start with backbone layer zero")
            self.reset(u.shape[0])
        if u.shape[0] != self._batch_size or index != self._expected_layer:
            raise RuntimeError("Batch shape or backbone layer order changed without a reset")
        length = u.shape[1]
        if index == 0:
            if len({entry.tokens for entry in self._cache}) != 1:
                raise RuntimeError("Per-layer cache positions disagree; reset the request")
            self._call_length = length
        elif length != self._call_length:
            raise RuntimeError("Backbone layers received different token counts")
        entry = self._cache[index]
        try:
            is_step = entry.tokens > 0 and length == 1
            # Native step uses rank-two linear/norm inputs; retain those GEMM
            # and adapter-hook shapes as well as its convolution arithmetic.
            projected = mx.in_proj(u.squeeze(1) if is_step else u)
            zxbcdt = projected.unsqueeze(1) if is_step else projected
            z, xbc, dt = torch.split(zxbcdt, (8192, 10240, 128), dim=-1)
            xbc = self._convolution(mx, xbc, entry)
            x, bm, cm = torch.split(xbc, (8192, 1024, 1024), dim=-1)
            batch = u.shape[0]
            x = x.reshape(batch, length, 128, 64)
            bm = bm.reshape(batch, length, 8, 128)
            cm = cm.reshape(batch, length, 8, 128)
            aa = -torch.exp(mx.A_log.float())
            y, stats, scores = readout_collect_scan(x, dt, aa, bm, cm, mx.D, mx.dt_bias, entry.state)
            if y.dtype != torch.float16 or tuple(y.shape) != (batch, length, 128, 64):
                raise RuntimeError("State codec returned unexpected readout dtype/shape")
            if self.collect_stats:
                self._accumulate_statistics(index, stats, batch, length)
                self._accumulate_readout(index, scores, batch, length)
            # Keep this module call intact: the external Resurface norm prehook
            # adds its post-D correction here, using the matching mixer input.
            y = y.reshape(batch, length, 8192)
            if is_step:
                y = mx.norm(y[:, 0], z[:, 0])
                out = mx.out_proj(y).unsqueeze(1)
            else:
                y = mx.norm(y, z)
                out = mx.out_proj(y)
            entry.tokens += length
            self._expected_layer = (index + 1) % len(self._mixers)
            if self._expected_layer == 0:
                self._call_length = None
            return out
        except Exception:
            self._failed = True
            raise

