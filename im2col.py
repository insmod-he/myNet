import numpy as np
import pdb

def im2col(X, pad, patchH, patchW):
    assert patchH==patchW
    assert 1==(patchH%2)
    assert 1==(patchW%2)

    X_pad = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant') # batch,channel,height,width
    batch,chn,height,width = X_pad.shape

    outH = height - patchH + 1
    outW = width  - patchW + 1
    
    # [patchH*patchW*chn, batch*outH*outW]
    X_cols = np.zeros( [patchH*patchW*chn, batch*outH*outW], dtype=np.float)
    
    #pdb.set_trace()
    for c in xrange(batch):
        for out_h in xrange(outH):
            for out_w in xrange(outW):
                patch = X_pad[c, :, out_h:(out_h+patchH), out_w:(out_w+patchW)]
                X_cols_idx = c*outH*outW + out_h*outW + out_w 
                X_cols[:, X_cols_idx] = patch.reshape(patch.size)
    return X_cols

def col2im(X_cols, pad, patchH, patchW, outH, outW):
    assert patchH==patchW
    assert 1==(patchH%2)
    assert 1==(patchW%2)

    #pdb.set_trace()
    patch_size,num = X_cols.shape        # path_size, bath*outH,outW
    dim   = patch_size / (patchH*patchW)
    batch = num/outH/outW

    Out = X_cols.reshape(patch_size, batch, outH, outW)
    ori_H = outH + patchH - 1
    ori_W = outW + patchH - 1
    ori_data = np.zeros( [batch, dim, ori_H+2*pad, ori_W+2*pad], dtype=np.float)

    hH = patchH/2
    hW = patchW/2
    for b in xrange(batch):
        for h in xrange(outH):
            for w in xrange(outW):
                patch = Out[:,b,h,w]
                patch = patch.reshape( [dim, patchH, patchW] )
                y = h
                x = w
                ori_data[b, :, y:(y+patchH), x:(x+patchW)] += patch
    if pad>0:
        ori_data = ori_data[:,:, pad:-pad, pad:-pad]
    return ori_data
        
    