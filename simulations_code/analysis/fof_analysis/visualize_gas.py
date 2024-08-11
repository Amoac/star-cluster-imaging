import numpy as np

#Making the function

def compute_velocity_array(xcen, ycen, zcen, xg, yg, zg, mg, hg, stepn = 16, pc_width = 25, nkernel = 200, kernel_file_path = '/home1/09528/amoac/', stepnh = 7.5):
    kpc_width = pc_width/1e3
    sizebox = kpc_width
    res = sizebox/stepn
    
    kernelfile = kernel_file_path + 'kernel2d'
    w2d_kernel = []
    w2d_table = np.zeros(nkernel+3)
    temp = open(kernelfile,'r').read().split('\n')
    

    for i in range(len(temp)-1):
        if (i%2 == 0):
            w2d_kernel.append(float(temp[i]))

    w2d_kernel.append(0.)
    w2d_kernel = np.array(w2d_kernel)

    
    w2darray = np.zeros(shape=(stepn, stepn))
    vel_array = np.zeros(shape=(stepn, stepn))
    ngas = len(ig[0])
    # Set the kernel weights, make sure to normalize properly
    htot = 0
    for i in range(ngas):
        htot += hg[i] / ngas
       

    # Big loop
    for i in range(ngas):
        xi = (xg[i] - xcen) / res + stepnh
        yi = (yg[i] - ycen) / res + stepnh
        hi = 2 * hg[i] / res
        
        ixmin = int(np.floor(xi - hi)) + 1
        ixmin = max(ixmin, 0)
        ixmin = min(ixmin, stepn - 1)

        ixmax = int(np.floor(xi + hi))
        ixmax = max(ixmax, 0)
        ixmax = min(ixmax, stepn - 1)

        iymin = int(np.floor(yi - hi)) + 1
        iymin = max(iymin, 0)
        iymin = min(iymin, stepn - 1)

        iymax = int(np.floor(yi + hi))
        iymax = max(iymax, 0)
        iymax = min(iymax, stepn - 1)

        w2d_sum = 0.
        d2max = 4. * hg[i] * hg[i]
        kernfrac = nkernel * 0.5 / hg[i]
        loop = np.arange(ixmin, ixmax + 1, 1)
        for ii in range(len(loop)):
            r0xx = (ii - stepnh) * res + xcen
            for jj in range(len(loop)):
                r0yy = (jj - stepnh) * res + ycen
                xx = xg[i] - r0xx
                yy = yg[i] - r0yy
                d2 = xx * xx + yy * yy
                if d2 <= d2max:
                    d = np.sqrt(d2)
                    xii = d * kernfrac
                    xxi = int(np.floor(xii))
                    w2d = (w2d_table[xxi + 1] - w2d_table[xxi]) * (xii - xxi) + w2d_table[xxi]
                    w2darray[ii, jj] = w2d
                    w2d_sum += w2d
                else:
                    w2darray[ii, jj] = 0.

        if w2d_sum > 0:
            weight = mg[i] / w2d_sum
            for ii in range(len(loop)):
                for jj in range(len(loop)):
                    if w2darray[ii, jj] > 0:
                        vel_array[ii, jj] += w2darray[ii, jj] * weight
        else:
            ii = int(np.floor(xi + 0.5))
            jj = int(np.floor(yi + 0.5))
            if ((ii >=0) & (ii <= stepn-1) & (jj >= 0) & (jj <= stepn-1)):
                vel_array[ii,jj] = vel_array[ii,jj] + mg[i]
        
        
    #return vel_array[0,:,:]
    return vel_array