# cython: boundscheck=False
# cython: wraparound=False
import numpy as np

def forward_step_explicit(double k_00, double k_01, double k_10, double increment):
	return (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)

def forward_step_implicit(double k_00, double k_01, double k_10, double increment):
	return k_01+k_10-k_00 + ((0.5*increment)/(1.-0.25*increment))*(k_01+k_10)

def sig_kernels_mmd(double[:,:,:] x, double[:,:,:] y, int n=0, bint sym=False,bint implicit=True):

	cdef int A = x.shape[0]
	cdef int B = y.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:,:] K = np.zeros((A,B,MM+1,NN+1), dtype=np.float64)

	if sym:
		for l in range(A):
			for m in range(l,A):

				for i in range(MM+1):
					K[l,m,i,0] = 1.
					K[m,l,i,0] = 1.
	
				for j in range(NN+1):
					K[l,m,0,j] = 1.
					K[m,l,0,j] = 1.

				for i in range(MM):
					for j in range(NN):

						ii = int(i/(2**n))
						jj = int(j/(2**n))

						increment = 0.
						for k in range(D):
							increment += (x[l,ii+1,k]-x[l,ii,k])*(y[m,jj+1,k]-y[m,jj,k])/factor
                        if implicit:
				            K[l,m,i+1,j+1] = forward_step_implicit(K[l,m,i,j], K[l,m,i,j+1], K[l,m,i+1,j], increment)
                        else:
                            K[l,m,i+1,j+1] = forward_step_explicit(K[l,m,i,j], K[l,m,i,j+1], K[l,m,i+1,j], increment)
                        K[m,l,i+1,j+1] = K[l,m,i+1,j+1]

	else:
		for l in range(A):
			for m in range(B):

				for i in range(MM+1):
					K[l,m,i,0] = 1.
	
				for j in range(NN+1):
					K[l,m,0,j] = 1.

				for i in range(MM):
					for j in range(NN):

						ii = int(i/(2**n))
						jj = int(j/(2**n))

						increment = 0.
						for k in range(D):
							increment += (x[l,ii+1,k]-x[l,ii,k])*(y[m,jj+1,k]-y[m,jj,k])/factor
                        if implicit:
						    K[l,m,i+1,j+1] = forward_step_implicit(K[l,m,i,j], K[l,m,i,j+1], K[l,m,i+1,j], increment)
                        else:
                            K[l,m,i+1,j+1] = forward_step_explicit(K[l,m,i,j], K[l,m,i,j+1], K[l,m,i+1,j], increment)
	return K