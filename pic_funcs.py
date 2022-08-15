# FFT solvers

def kn(N, L):
    '''
    Gets array of wavenumbers with k = n*(2pi/L)
    
    Parameters
    ----------
    N: `int`
        number of cells in the grid
    L: `float`
        total length of system, including the last point
    
    Returns
    -------
    kn: `numpy.ndarray`
        wavenumbers
    '''
    # going from -N / 2 to N / 2 in integer steps
    n_min = np.floor(-N / 2)
    n_max = np.ceil(N / 2)
    n = np.arange(n_min, n_max, 1)
    kn = n * 2 * np.pi / L
    
    return(kn)

def FFT_1D(gx, x):
    '''
    Discrete fourier transform.
    
    Parameters
    ----------
    gx: `np.ndarray` 
        grid-function of position x (1d arr)
    x: `np.ndarray`
        positions
        
    Returns
    -------
    gx: `np.ndarray`
        fft of gx -> gk
    '''
    dx = x[1] - x[0] # length of a cell
    L = x[-1] + dx # total length of system, including the last point
    N = len(x) # number of cells
    
    k = kn(N, L)
    gk = np.zeros((N), dtype=np.complex_)
    # summing over the wavenumber
    for i in range(N):
        gk[i] = dx * np.trapz(gx * np.exp(- 1j * k[i] * x)) 
    
    return (gk, k)

def IFFT_1D(gk, k, x):
    '''
    inverse fourier transform
    
    Parameters
    ----------
    gk: `np.ndarray`
        function in k space
    k: `np.ndarray`
        wavenumbers
    x: `np.ndarray`
        positions
    
    Returns
    -------
    gx: `np.ndarray`
        function in position space
    '''
    dx = x[1]-x[0] # length of a cell
    L = x[-1]+dx # total length of system, including the last point
    N = len(x) # number of cells
    
    gx = np.zeros((N), dtype=np.complex_)
    
    # summing over x
    for i in range(N):
        # using np.nansum because `gk` will have a nan value at k = 0
        gx[i] = (1 / L) * np.nansum(gk * np.exp(1j * k * x[i])) 
    
    return gx

def solve_tri_matrix(tridiag_matrix, d_arr):   
    N = len(tridiag_matrix)
    # getting diagonal elements from right diagonal
    c_arr = np.diag(tridiag_matrix, 1)
    c_arr = np.append(c_arr, 0) # inserting 0 in nth position (c_n = 0)
    
    # getting diagonal elements from left diagonal
    a_arr = np.diag(tridiag_matrix, -1)
    a_arr = np.insert(a_arr, 0, 0) # inserting 0 in first position (a_0 = 0)
    
    # getting diagonal elements from center diagonal
    b_arr = np.diag(tridiag_matrix, 0)
    
    c_sweep = np.zeros(N)
    d_sweep = np.zeros(N)
    c_sweep[0] = c_arr[0] / b_arr[0]
    d_sweep[0] = d_arr[0] / b_arr[0]
    for i in range(1, N):
        c_sweep[i] = c_arr[i] / (b_arr[i] - a_arr[i] * c_sweep[i - 1])
        d_sweep[i] = (d_arr[i] - a_arr[i] * d_sweep[i - 1]) \
                        / (b_arr[i] - a_arr[i] * c_sweep[i - 1])

    x_arr = np.zeros(N)
    x_arr[N - 1] = d_sweep[N - 1]
    for i in range(N - 2, -1, -1):
        # back substitution
        x_arr[i] = d_sweep[i] - c_sweep[i] * x_arr[i + 1]
        
    return x_arr

def solve_almost_tri(almost_tri_matrix, d_arr, gamma=1):
    n = len(almost_tri_matrix)
    c_n = almost_tri_matrix[-1][0]
    a_1 = almost_tri_matrix[0][-1]
    
    B = almost_tri_matrix.copy()
    B[0][0] -= gamma
    B[-1][-1] -= c_n * a_1 / gamma
    B[0][-1] = 0
    B[-1][0] = 0
    
    u = np.zeros(n)
    u[0] = gamma
    u[-1] = c_n
    
    v = np.zeros(n)
    v[0] = 1
    v[-1] = a_1 / gamma
    y = solve_tri_matrix(B, d_arr)
    q = solve_tri_matrix(B, u)
    
    val = (v[0] * y[0] + v[-1] * y[-1]) / (1 + v[0] * q[0] + v[-1] * q[-1])
    x = y - q * val
    return x