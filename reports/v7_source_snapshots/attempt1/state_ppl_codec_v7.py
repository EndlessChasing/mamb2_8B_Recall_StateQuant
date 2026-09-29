#!/usr/bin/env python3
"""Actual packed v7 nonuniform INT4 state codebooks, with unchanged INT8.

Same52B row:16 signed INT8,64 signed codebook indices,48 zero-carry,2 FP16
scales. Constants compile into arithmetic; no resident codebook/LUT is allocated.
Current readout precedes per-token carry quantization. Uniform delegates v6.
Derived from Apache-2.0 state_codec.py; see docs/THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
from pathlib import Path
import struct
import sys
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall import state_codec as legacy_codec
from state_ppl_codec_v6 import StatePPLQuant, _stored_quantize, validate_finite_result
from state_ppl_codec_v6 import scan as scan_v6

PROTOCOL_SHA = '84ffdce1d9e5d5dbac2b6996072f0db09bd1bf79fd3c55dad8b8f35973cd5d14'
RP = 16

def fp32(value): return struct.unpack('<f', struct.pack('<f', value))[0]
CODEBOOKS = {
    'uniform': tuple(float(i) for i in range(8)),
    'mild': (0., .5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.),
    'quadratic': tuple(fp32(i*i/7.) for i in range(8)),
    'fp4like': tuple(fp32(v) for v in (0., 7/12, 7/6, 7/4, 7/3, 7/2, 14/3, 7.)),
}
CODEBOOK_BITS = {name: [struct.unpack('<I', struct.pack('<f', v))[0] for v in levels]
                 for name, levels in CODEBOOKS.items()}


def validate_codebook(codebook):
    if codebook not in CODEBOOKS: raise ValueError('Unknown frozen v7 codebook')


def allocate_state(batch, heads, dim, dstate, device):
    return legacy_codec.allocate_state(batch, heads, dim, dstate, 'sq3p25', device)


@triton.jit
def _decode4(q, scale, LEVELS: tl.constexpr):
    magnitude = tl.full(q.shape, 0., tl.float32)
    for i in tl.static_range(1, 8):
        magnitude = tl.where(tl.abs(q) == i, tl.full((), LEVELS[i], tl.float32), magnitude)
    # Force actual FP32 decoded carry before recurrence; never fuse this product.
    decoded = libdevice.mul_rn(magnitude, scale)
    return tl.where(q < 0, -decoded, decoded)


@triton.jit
def _quantize4(value, scale, LEVELS: tl.constexpr):
    # Adjacent decoded FP32 levels have exponent span <=3 (or a zero endpoint),
    # so their FP64 sum/half is exact. Comparison chooses the nearest ACTUAL
    # FP32 reconstruction; equality selects the larger magnitude.
    absolute = tl.abs(value).to(tl.float64)
    previous = tl.full(scale.shape, 0., tl.float32)
    index = tl.full(value.shape, 0, tl.int32)
    for i in tl.static_range(1, 8):
        current = libdevice.mul_rn(tl.full((), LEVELS[i], tl.float32), scale)
        midpoint = (previous.to(tl.float64) + current.to(tl.float64)) * .5
        index += (absolute >= midpoint).to(tl.int32)
        previous = current
    signed = tl.where(value < 0., -index, index)
    return tl.where(scale > 0., signed, 0).to(tl.int32)


@triton.jit
def _probe(X, S, Q, R, COUNT: tl.constexpr, LEVELS: tl.constexpr):
    i = tl.program_id(0)*128+tl.arange(0,128)
    x = tl.load(X+i,mask=i<COUNT,other=0.).to(tl.float32)
    s = tl.load(S+i,mask=i<COUNT,other=0.).to(tl.float32)
    q = _quantize4(x,s,LEVELS)
    tl.store(Q+i,q,mask=i<COUNT)
    tl.store(R+i,_decode4(q,s,LEVELS),mask=i<COUNT)


def quantize_probe(values, scales, codebook):
    validate_codebook(codebook)
    if codebook == 'uniform': raise ValueError('Uniform uses frozen v6 stored-scale quantizer')
    if (values.dtype != torch.float32 or scales.dtype != torch.float16 or not values.is_cuda
            or values.shape != scales.shape or values.device != scales.device):
        raise ValueError('Expected CUDA FP32 values and same-shape FP16 scales')
    values,scales=values.contiguous(),scales.contiguous()
    q=torch.empty_like(values,dtype=torch.int32); decoded=torch.empty_like(values)
    _probe[(triton.cdiv(values.numel(),128),)](values,scales,q,decoded,values.numel(),CODEBOOKS[codebook])
    return q,decoded

@triton.jit
def _nonuniform_scan(X, DT, A, B, C, D, DB, Perm, Lo, Hi, Q4, S8, S4, Y, L: tl.constexpr, H: tl.constexpr, P: tl.constexpr, G: tl.constexpr, TILE: tl.constexpr, LEVELS: tl.constexpr):
    b, h, tile = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    p = tile * TILE + tl.arange(0, TILE)
    mp = p < P
    row = (b * H + h) * P + p
    n8 = tl.arange(0, 16)
    n4 = tl.arange(0, 64)
    nd = tl.arange(0, 64)
    lo = tl.load(Lo + row[:, None] * 8 + n8[None, :] // 2, mask=mp[:, None], other=0).to(tl.int32) >> n8[None, :] % 2 * 4 & 15
    hi = tl.load(Hi + row[:, None] * 8 + n8[None, :] // 2, mask=mp[:, None], other=0).to(tl.int32) >> n8[None, :] % 2 * 4 & 15
    ub = hi << 4 | lo
    q8 = ub - 256 * (ub > 127).to(tl.int32)
    nib = tl.load(Q4 + row[:, None] * 32 + n4[None, :] // 2, mask=mp[:, None], other=0).to(tl.int32) >> n4[None, :] % 2 * 4 & 15
    q4 = nib - 16 * (nib > 7).to(tl.int32)
    scale8 = tl.load(S8 + row, mask=mp, other=0).to(tl.float32)
    scale4 = tl.load(S4 + row, mask=mp, other=0).to(tl.float32)
    state8 = q8.to(tl.float32) * scale8[:, None]
    state4 = _decode4(q4, scale4[:, None], LEVELS)
    av = tl.load(A + h).to(tl.float32)
    dv = tl.load(D + h).to(tl.float32)
    bias = tl.load(DB + h).to(tl.float32)
    group = h // (H // G)
    pos8 = tl.load(Perm + group * 128 + n8).to(tl.int32)
    pos4 = tl.load(Perm + group * 128 + 16 + n4).to(tl.int32)
    posd = tl.load(Perm + group * 128 + 80 + nd, mask=nd < 48, other=0).to(tl.int32)
    for t in range(L):
        xoff = ((b * L + t) * H + h) * P + p
        x = tl.load(X + xoff, mask=mp, other=0).to(tl.float32)
        dt = tl.load(DT + (b * L + t) * H + h).to(tl.float32) + bias
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1.0), dt)
        decay = tl.exp(av * dt)
        bbase = ((b * L + t) * G + group) * 128
        bv8 = tl.load(B + bbase + pos8).to(tl.float32)
        cv8 = tl.load(C + bbase + pos8).to(tl.float32)
        bv4 = tl.load(B + bbase + pos4).to(tl.float32)
        cv4 = tl.load(C + bbase + pos4).to(tl.float32)
        bvd = tl.load(B + bbase + posd, mask=nd < 48, other=0).to(tl.float32)
        cvd = tl.load(C + bbase + posd, mask=nd < 48, other=0).to(tl.float32)
        state8 = state8 * decay + bv8[None, :] * dt * x[:, None]
        state4 = state4 * decay + bv4[None, :] * dt * x[:, None]
        stated = bvd[None, :] * dt * x[:, None]
        out = tl.sum(state8 * cv8[None, :], 1)
        out += tl.sum(state4 * cv4[None, :], 1)
        out += tl.sum(stated * cvd[None, :], 1) + x * dv
        tl.store(Y + xoff, out, mask=mp)
        den8 = tl.maximum(tl.div_rn(tl.max(tl.abs(state8), 1), 127.0), 1e-08)
        den4 = tl.maximum(tl.div_rn(tl.max(tl.abs(state4), 1), 7.0), 1e-08)
        scale8 = den8.to(tl.float16).to(tl.float32)
        scale4 = den4.to(tl.float16).to(tl.float32)
        q8 = _stored_quantize(state8, scale8[:, None], 127)
        q4 = _quantize4(state4, scale4[:, None], LEVELS)
        state8 = q8.to(tl.float32) * scale8[:, None]
        state4 = _decode4(q4, scale4[:, None], LEVELS)
    qlo = (q8 & 15).to(tl.uint8)
    qhi = (q8 >> 4 & 15).to(tl.uint8)
    lo0, lo1 = tl.split(tl.reshape(qlo, (TILE, 8, 2)))
    hi0, hi1 = tl.split(tl.reshape(qhi, (TILE, 8, 2)))
    pairs8 = tl.arange(0, 8)
    tl.store(Lo + row[:, None] * 8 + pairs8[None, :], lo0 | lo1 << 4, mask=mp[:, None])
    tl.store(Hi + row[:, None] * 8 + pairs8[None, :], hi0 | hi1 << 4, mask=mp[:, None])
    qn = (q4 & 15).to(tl.uint8)
    f0, f1 = tl.split(tl.reshape(qn, (TILE, 32, 2)))
    pairs4 = tl.arange(0, 32)
    tl.store(Q4 + row[:, None] * 32 + pairs4[None, :], f0 | f1 << 4, mask=mp[:, None])
    tl.store(S8 + row, scale8, mask=mp)
    tl.store(S4 + row, scale4, mask=mp)

@torch.no_grad()
def scan(x, dt, A, B, C, D, dt_bias, state, permutation, *, codebook='uniform'):
    validate_codebook(codebook)
    if codebook == 'uniform':
        return scan_v6(x,dt,A,B,C,D,dt_bias,state,permutation,scale_mode='stored_scale',int4_clip=1.)
    if x.ndim != 4 or x.dtype != torch.float16 or not x.is_cuda:
        raise ValueError('x must be CUDA FP16[batch,tokens,heads,dim]')
    batch, length, heads, dim = x.shape
    if length < 1 or state.shape != (batch, heads, dim, 128):
        raise ValueError('Positive sequence and matching state geometry required')
    legacy_codec._validate_state(state, x.device)
    if state.mode != 'sq3p25': raise ValueError('Packed SQ3.25 state required')
    if B.ndim != 4: raise ValueError('B must have4 axes')
    groups = B.shape[2]
    if groups < 1 or heads % groups: raise ValueError('Heads must be divisible by groups')
    shapes = ((dt, (batch, length, heads)), (A, (heads,)),
        (B, (batch, length, groups, 128)), (C, (batch, length, groups, 128)),
        (D, (heads,)), (dt_bias, (heads,)))
    for tensor, shape in shapes:
        if tuple(tensor.shape) != shape or tensor.device != x.device:
            raise ValueError('Input device/geometry differs')
    if A.dtype != torch.float32 or any(v.dtype != torch.float16 for v in (dt, B, C)):
        raise ValueError('A must be FP32; dt/B/C must be FP16')
    if any(v.dtype not in (torch.float16, torch.float32) for v in (D, dt_bias)):
        raise ValueError('D/dt_bias must be FP16 or FP32')
    if (permutation.dtype != torch.uint8 or tuple(permutation.shape) != (groups, 128)
            or permutation.device != x.device):
        raise ValueError('Permutation must be CUDA uint8[groups,128]')
    x, dt, A, B, C, D, dt_bias, permutation = (
        v.contiguous() for v in (x, dt, A, B, C, D, dt_bias, permutation))
    output = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _nonuniform_scan[(batch,heads,triton.cdiv(dim,RP))](x,dt,A,B,C,D,dt_bias,permutation,
            *(state.tensors[k] for k in ('lo','hi','q4','s8','s4')),output,
            length,heads,dim,groups,RP,CODEBOOKS[codebook],num_warps=4)
    return output,None


@torch.no_grad()
def decode_state(state, permutation=None, *, codebook):
    validate_codebook(codebook)
    values = legacy_codec.decode_state(state)
    if codebook != 'uniform':
        # Inspection helper only; temporary tensors are not request cache.
        q4=state.tensors['q4'].int()
        nibbles=torch.stack((q4 & 15,(q4 >> 4)&15),-1).flatten(-2)
        indices=torch.where(nibbles>7,nibbles-16,nibbles)
        magnitude=torch.zeros_like(indices,dtype=torch.float32)
        for i,level in enumerate(CODEBOOKS[codebook]):
            magnitude=torch.where(indices.abs()==i,level,magnitude)
        reconstructed=magnitude*state.tensors['s4'].float()[...,None]
        values[...,16:80]=torch.where(indices<0,-reconstructed,reconstructed)
    if permutation is None:return values
    groups=permutation.shape[0]
    legacy_codec.validate_permutation(permutation,groups)
    batch,heads,dim,_=state.shape
    if heads%groups:raise ValueError('Invalid permutation grouping')
    indices=permutation.long().repeat_interleave(heads//groups,0)[None,:,None,:].expand(batch,heads,dim,128)
    return torch.zeros_like(values).scatter_(-1,indices,values)


class StatePPLQuantV7(StatePPLQuant):
    def __init__(self, model, permutations, *, codebook='uniform'):
        validate_codebook(codebook)
        self.codebook=codebook
        super().__init__(model,permutations,scale_mode='stored_scale',int4_clip=1.)

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
            permutation = None if self.permutations is None else self.permutations[index]
            y, stats = scan(x, dt, aa, bm, cm, mx.D, mx.dt_bias, entry.state,
                            permutation=permutation, codebook=self.codebook)
            if y.dtype != torch.float16 or tuple(y.shape) != (batch, length, 128, 64):
                raise RuntimeError("State codec returned unexpected readout dtype/shape")
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

    def cache_breakdown(self):
        result=super().cache_breakdown()
        result.update(codebook=self.codebook,codebook_bits=CODEBOOK_BITS[self.codebook],resident_codebook_bytes=0,
            scope='Actual packed state, FP16 convolution and one permutation table. Codebook constants compile into arithmetic; no resident LUT. Weights, activations and allocator reserve excluded.')
        return result
