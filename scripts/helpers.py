import numpy as np
import xerus
from itertools import chain


def fermi_pasta_ulam(number_of_oscillators, number_of_snapshots):
    """Fermi–Pasta–Ulam problem.
    Generate data for the Fermi–Pasta–Ulam problem represented by the differential equation
        d^2/dt^2 x_i = (x_i+1 - 2x_i + x_i-1) + 0.7((x_i+1 - x_i)^3 - (x_i-x_i-1)^3).
    See [1]_ for details.
    Parameters
    ----------
    number_of_oscillators: int
        number of oscillators
    number_of_snapshots: int
        number of snapshots
    Returns
    -------
    snapshots: ndarray(number_of_oscillators, number_of_snapshots)
        snapshot matrix containing random displacements of the oscillators in [-0.1,0.1]
    derivatives: ndarray(number_of_oscillators, number_of_snapshots)
        matrix containing the corresponding derivatives
    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # define random snapshot matrix
    snapshots = 2 * np.random.rand(number_of_oscillators, number_of_snapshots) - 1

    # compute derivatives
    derivatives = np.zeros((number_of_oscillators, number_of_snapshots))
    for j in range(number_of_snapshots):
        derivatives[0, j] = snapshots[1, j] - 2 * snapshots[0, j] + 0.7 * (
                (snapshots[1, j] - snapshots[0, j]) ** 3 - snapshots[0, j] ** 3)
        for i in range(1, number_of_oscillators - 1):
            derivatives[i, j] = snapshots[i + 1, j] - 2 * snapshots[i, j] + snapshots[i - 1, j] + 0.7 * (
                    (snapshots[i + 1, j] - snapshots[i, j]) ** 3 - (snapshots[i, j] - snapshots[i - 1, j]) ** 3)
        derivatives[-1, j] = - 2 * snapshots[-1, j] + snapshots[-2, j] + 0.7 * (
                -snapshots[-1, j] ** 3 - (snapshots[-1, j] - snapshots[-2, j]) ** 3)

    return snapshots, derivatives

def basis(b):
    """Different basis functions
    Parameters
    ----------
    b: int
        which basisfunction
      
    Returns
    -------
    psi: list of functions
        basis functions based on b
    """
    if b == 0:
        print("L2(-1,1) orthogonal polynomials")
        psi = [lambda t: 1, lambda t: t, lambda t: 0.5*(3*t ** 2-1), lambda t: 0.5* (5*t ** 3-3*t)]
    elif b == 1:
        print("H1(-1,1) orthogonal polynomials, H1 normalized")
        psi = [lambda t: 1/np.sqrt(2),\
               lambda t: np.sqrt(3/8)*t,\
               lambda t: np.sqrt(45/128) *t ** 2\
                    -np.sqrt(5/128),\
               lambda t: 5*np.sqrt(7/302) *t ** 3\
                    -9/2*np.sqrt(7/302) *t]
    elif b == 2:
        print("monomials")
        psi = [lambda t: 1, lambda t: t, lambda t: t ** 2, lambda t: t ** 3]
    return psi

def random_data_selection(Alist,C1ex,C2list,noo,nos):    
    """Creates label data for the given solution C1ex
    Parameters
    ----------
    Alist: list of xerus tensors
        Dictionary in CP format
    C1ex: xerus TTOperator
        Solution for which samples shall be constructed
    C2list: list of xerus tensors
        Selection operator in CP format
    noo: int
        number of dimensions
    nos: int
        number of samples
      
    Returns
    -------
    Y: numpy array
        nos times noo matrix containing the labels for the given solution
    """
    i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3,k4 = xerus.indices(14)
    l = np.ones([nos,1,noo])
    r = np.ones([nos,1,noo])
    tmp1 = xerus.Tensor()
    for i in range(noo):
        t1 = C1ex.get_component(i)
        t2 = C2list[i]
        t3 = Alist[i]
        tmp1(i1,i2,i3,i4) <<  t3(k1,i2)* t1(i1,k1,k2,i4)* t2(k2,i3)
        tmp1np = tmp1.to_ndarray()
        l = np.einsum('mid,imdj->mjd',l,tmp1np)
    return np.einsum('mid,mid->dm',l,r)


def build_data_tensor_list2(noo,x,nos,psi,p):
    """Creates the dictionary for given basis functions and samples
    Parameters
    ----------
    noo: int
        number of dimensions
    x: numpy array
        nooxnos containing the sample data
    nos: int
        number of samples
    psi: list of functions
        basisfunctions
    p: int
        number of basisfunctions   
    Returns
    -------
    Alist: list of xerus tensor
        Dictionary operator in CP format
    """
    AList = []
    for i in range(noo):
        tmp = xerus.Tensor([p,nos])
        for k in range(nos):
            for l in range(p):
                tmp[l,k] = psi[l](x[i,k])  
        AList.append(tmp)       
    return AList


def build_choice_tensor2(noo): 
    """Creates the Selection Tensor for the Fermi Pasta activation pattern
    Parameters
    ----------
    noo: int
        number of dimensions
    Returns
    -------
    Alist: list of xerus tensor
        Selection tensor in CP format for the Fermi Pasta activation pattern
    """
    C2list = []
    tmp = xerus.Tensor([3,noo])
    for j in range(noo):
        if j==0:
            tmp[2,j] = 1
        elif j==1:
            tmp[1,j] = 1
        else:
            tmp[0,j] = 1
    C2list.append(tmp)
    for i in range(1,noo-1):
        tmp = xerus.Tensor([4,noo])   
        for j in range(noo):
            if j==i-1:
                tmp[3,j] = 1
            elif j==i:
                tmp[2,j] = 1
            elif j==i+1:
                tmp[1,j] = 1
            else:
                tmp[0,j] = 1
        C2list.append(tmp)
    
    tmp = xerus.Tensor([3,noo])   
    for j in range(noo):
        if j==noo - 2:
            tmp[2,j] = 1
        elif j==noo - 1:
            tmp[1,j] = 1
        else:
            tmp[0,j] =1  
    C2list.append(tmp)

    return C2list

def construct_exact_fermit_pasta_single_TT(noo,p,beta):
    """Creates exact solution in single TT format for the Fermi Pasta problems in the monomials basis,
    for other basis functions this needs to be transformed!
    Parameters
    ----------
    noo: int
        number of dimensions
    p: int
        number of basis functions
    beta: float
        coefficient of the fermit pasta equation
    Returns
    -------
    Solution: xerus TTOperator
        Exact soulution of FPTU problem in monomials basis functions
    """
    dim = [p for i in range(0,noo)]
    dim.append(noo)
    Solution = xerus.TTTensor(dim)

    tmp = xerus.Tensor([1,4,4*noo])
    for eq in range(noo):
        tmp[0,0,4*eq] = 1
    tmp[0,0,0] = 0
    tmp[0,1,0] = -2 
    tmp[0,3,0] = -2*beta
    tmp[0,0,1] = 1
    tmp[0,2,1] = 3*beta
    tmp[0,1,2] = -3*beta
    tmp[0,0,3] = beta

    tmp[0,0,4] = 1
    tmp[0,1,5] = 1
    tmp[0,2,6] = 1
    tmp[0,3,7] = 1
    Solution.set_component(0,tmp)


    for comp in range(1,Solution.order()-1):
        tmp = xerus.Tensor([4*noo,4,4*noo])
        for eq in range(noo):
            tmp[4*eq,0,4*eq] = 1
        if (comp+1)*4 < 4*noo:
            tmp[4*(comp+1),0,4*(comp+1)] = 1
            tmp[4*(comp+1),1,4*(comp+1)+1] = 1
            tmp[4*(comp+1),2,4*(comp+1)+2] = 1
            tmp[4*(comp+1),3,4*(comp+1)+3] = 1

        tmp[4*comp,0,4*comp] = 0
        tmp[4*comp,1,4*comp] = -2 
        tmp[4*comp,3,4*comp] = -2*beta
        tmp[4*comp+1,0,4*comp] = 1       
        tmp[4*comp+1,2,4*comp] = 3*beta
        tmp[4*comp+2,1,4*comp] = -3*beta
        tmp[4*comp+3,0,4*comp] = beta
        tmp[4*comp,0,4*comp+1] = 1       
        tmp[4*comp,2,4*comp+1] = 3*beta
        tmp[4*comp,1,4*comp+2] = -3*beta
        tmp[4*comp,0,4*comp+3] = beta

        tmp[4*(comp-1),0,4*(comp-1)] = 1
        tmp[4*(comp-1)+1,1,4*(comp-1)] = 1
        tmp[4*(comp-1)+2,2,4*(comp-1)] = 1
        tmp[4*(comp-1)+3,3,4*(comp-1)] = 1

        Solution.set_component(comp,tmp)

    tmp = xerus.Tensor([4*noo,noo,1])
    for eq in range(noo):
        tmp[4*eq,eq,0] = 1
    Solution.set_component(Solution.order()-1,tmp)
    Solution.round(0.0)
    return Solution


def construct_exact_fermit_pasta(noo,p,beta):
    """Creates exact solution in selection format for the Fermi Pasta problems in the monomials basis,
    for other basis functions this needs to be transformed!
    Parameters
    ----------
    noo: int
        number of dimensions
    p: int
        number of basis functions
    beta: float
        coefficient of the fermit pasta equation
    Returns
    -------
    C1ex: xerus TTOperator
        Exact soulution of FPTU problem in monomials basis functions
    """
    s = 3
    rank = 4
    dim = [p for i in range(0,noo)]
    dim.extend([s+1 for i in range(0,noo)])
    dim[noo] = s
    dim[2*noo-1]=s
    C1ex = xerus.TTOperator(dim)

    comp = xerus.Tensor([rank,p,s+1,rank])
    #s=0
    comp[0,0,0,0] = 1
    #s=1
    comp[0,0,1,0] = 1
    comp[0,1,1,1] = 1
    comp[0,2,1,2] = 1
    comp[0,3,1,3] = 1
    #s=2
    comp[0,1,2,0] = -2
    comp[0,3,2,0] = -2*beta
    comp[0,2,2,1] = 3*beta
    comp[0,0,2,1] = 1
    comp[0,1,2,2] = -3*beta
    comp[0,0,2,3] = beta

    comp[1,2,2,0] = 3*beta
    comp[1,0,2,0] = 1
    comp[2,1,2,0] = -3*beta
    comp[3,0,2,0] = beta
    #s=3
    comp[0,0,3,0] = 1
    comp[1,1,3,0] = 1
    comp[2,2,3,0] = 1
    comp[3,3,3,0] = 1

    comp0 = xerus.Tensor([1,p,s,rank])
    #s=0
    comp0[0,0,0,0] = 1
    #s=1
    comp0[0,0,1,0] = 1
    comp0[0,1,1,1] = 1
    comp0[0,2,1,2] = 1
    comp0[0,3,1,3] = 1
    #s=2
    comp0[0,1,2,0] = -2
    comp0[0,3,2,0] = -2*beta
    comp0[0,2,2,1] = 3*beta
    comp0[0,0,2,1] = 1
    comp0[0,1,2,2] = -3*beta
    comp0[0,0,2,3] = beta

    compd = xerus.Tensor([rank,p,s,1])
    #s=0
    compd[0,0,0,0] = 1
    #s=1
    compd[0,1,1,0] = -2
    compd[0,3,1,0] = -2*beta
    compd[1,2,1,0] = 3*beta
    compd[1,0,1,0] = 1
    compd[2,1,1,0] = -3*beta
    compd[3,0,1,0] = beta
    #s=2
    compd[0,0,2,0] = 1
    compd[1,1,2,0] = 1
    compd[2,2,2,0] = 1
    compd[3,3,2,0] = 1

    C1ex.set_component(0,comp0)
    for i in range(1,noo-1):
        C1ex.set_component(i,comp)
    C1ex.set_component(noo-1,compd)
    return C1ex

def construct_exact_fermit_pasta_random(noo,p, mean=False):
    """Creates exact solution in selection format for the Fermi Pasta problems 
    with random beta_i in the monomials basis with mean field,
    for other basis functions this needs to be transformed!
    Parameters
    ----------
    noo: int
        number of dimensions
    p: int
        number of basis functions
    mean: bool
        if mean field should be used
    Returns
    -------
    C1ex: xerus TTOperator
        Exact soulution of FPTU problem in monomials basis functions
    m: np.array
        the mean field coefficients
    beta: the bet coeffcients
    """
    s = 3
    rank = 4
    dim = [p for i in range(0,noo)]
    dim.extend([s+1 for i in range(0,noo)])
    dim[noo] = s
    dim[2*noo-1]=s
    C1ex = xerus.TTOperator(dim)

    beta = 2 * np.random.rand(noo) - 1
    if mean:
        m = 2 * np.random.rand(noo) - 1
    else:
        m = np.zeros(noo) 

    for i in range(1,noo-1):
        comp = xerus.Tensor([rank,p,s+1,rank])
        #s=0
        comp[0,0,0,0] = 1
        comp[0,1,0,1] = m[i]
        comp[1,0,0,1] = 1        
        #s=1
        comp[0,0,1,0] = 1
        comp[0,1,1,1] = 1
        comp[0,2,1,2] = 1
        comp[0,3,1,3] = 1
        comp[1,0,1,3] = 1/beta[i+1]

        #s=2
        comp[0,1,2,0] = -2 + m[i]
        comp[0,3,2,0] = -2*beta[i]
        comp[0,2,2,1] = 3*beta[i]
        comp[0,0,2,1] = 1 + m[i+1]
        comp[0,1,2,2] = -3*beta[i]
        comp[0,0,2,3] = beta[i]

        comp[1,2,2,0] = 3*beta[i]
        comp[1,0,2,0] = 1 + m[i-1]
        comp[2,1,2,0] = -3*beta[i]
        comp[3,0,2,0] = beta[i]

        #s=3
        comp[3,0,3,0] = 1/beta[i-1]
        comp[0,0,3,1] = 1
        comp[1,1,3,1] = 1
        comp[2,2,3,1] = 1
        comp[3,3,3,1] = 1

        C1ex.set_component(i,comp)

    comp0 = xerus.Tensor([1,p,s,rank])
    #s=0
    comp0[0,0,0,0] = 1
    comp0[0,1,0,1] = m[0]    
    #s=1
    comp0[0,0,1,0] = 1
    comp0[0,1,1,1] = 1
    comp0[0,2,1,2] = 1
    comp0[0,3,1,3] = 1

    #s=2
    comp0[0,1,2,0] = -2 + m[0]
    comp0[0,3,2,0] = -2*beta[0]
    comp0[0,2,2,1] = 3*beta[0]
    comp0[0,0,2,1] = 1  + m[1]
    comp0[0,1,2,2] = -3*beta[0]
    comp0[0,0,2,3] = beta[0]


    compd = xerus.Tensor([rank,p,s,1])
    #s=0
    compd[0,1,0,0] = m[noo-1]
    compd[1,0,0,0] = 1    
    #s=1
    compd[0,1,1,0] = -2 + m[noo-1]
    compd[0,3,1,0] = -2*beta[noo-1]
    compd[1,2,1,0] = 3*beta[noo-1]
    compd[1,0,1,0] = 1 + m[noo-2]
    compd[2,1,1,0] = -3*beta[noo-1]
    compd[3,0,1,0] = beta[noo-1]
    #s=2
    compd[0,0,2,0] = 1
    compd[1,1,2,0] = 1
    compd[2,2,2,0] = 1
    compd[3,3,2,0] = 1

    C1ex.set_component(0,comp0)
    
    C1ex.set_component(noo-1,compd)
    return C1ex,m,beta


def run_als(noo,nos,C1,C2list,Alist,C1ex,Y,max_iter,lam):
    """Perform the regularized ALS simulation
    Parameters
    ----------
    noo: int
        number of dimension
    nos: int
        number of samples
    C1: xerus TTTensor
        iterate tensor
    C2list: list of xerus tensor
        selection tensor
    Alist: list of xerus tensor
        dictionary tensor
    C1ex: xerus TTTensor
        exact solution
    Y: xerus Tensor
        right hand side
    max_iter: int
        number of iterations
    lam: float
        regularization parameter      
    Returns
    -------
    errors: list of floats
        list of relative error after each sweep
    """
    i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3,k4 = xerus.indices(14)
    diff = C1ex-C1
    tmp1 = xerus.Tensor()
    l = np.ones([noo,1,1,noo])
    r = np.ones([noo,1,1,noo])
    for i in range(noo):
        t1 = diff.get_component(i)
        t2 = C2list[i]
        tmp1(i1,i2,i3,j1,j2,j3) <<  t2(k1,i3) * t1(i1,k2,k1,j1) * t1(i2,k2,k3,j2)* t2(k3,j3)
        tmp1np = tmp1.to_ndarray()
        l = np.einsum('dije,ijdkle->dkle',l,tmp1np) 
    lr = np.sqrt(np.einsum('ijkl,ijkl->',l,r)) / C1ex.frob_norm()
    errors = [lr]
    errors2 = [1]
    lams = [lam]
    # Initialize stacks
    rStack = [xerus.Tensor.ones([noo,1,nos])]
    lStack = [xerus.Tensor.ones([noo,1,nos])]
    for ind in range(noo-1,0,-1):
        C1_tmp = C1.get_component(ind)
        C2_tmp = C2list[ind]
        A_tmp = Alist[ind]
        C1_tmp(i1,i2,i3,i4) << C1_tmp(i1,k1,k2,i4) * A_tmp(k1,i2) *C2_tmp(k2,i3)
        rstacknp = rStack[-1].to_ndarray()
        C1_tmpnp = C1_tmp.to_ndarray()
        rstacknpres = np.einsum('imdj,djm->dim',C1_tmpnp,rstacknp)  
        rStack.append(xerus.Tensor.from_ndarray(rstacknpres))
    
    forward = True
    mem = -1
    for it in range(0,max_iter):
        for pos in chain(range(0,noo), range(noo-1,-1,-1)):
            if mem == pos:
                forward = not forward
            op = xerus.Tensor()
            rhs = xerus.Tensor()
            C2i = C2list[pos]
            Ai = Alist[pos]
            Ainp = Ai.to_ndarray()
            C2inp = C2i.to_ndarray()


            lStacknp = lStack[-1].to_ndarray()
            rStacknp = rStack[-1].to_ndarray()


            op_pre_np = np.einsum('dim,pm,sd,djm->ipsjmd',lStacknp,Ainp,C2inp,rStacknp)
            op_pre = xerus.Tensor.from_ndarray(op_pre_np)


            op(i1,i2,i3,i4,j1,j2,j3,j4) << op_pre(i1,i2,i3,i4,k1,k2) * op_pre(j1,j2,j3,j4,k1,k2)
            
            rhs(i1,i2,i3,i4) <<  op_pre(i1,i2,i3,i4,k1,k2) * Y(k2,k1)
            op += lam * xerus.Tensor.identity(op.dimensions)

            op_arr = op.to_ndarray()
            rhs_arr = rhs.to_ndarray()

            op_dim = op.dimensions
            op_arr_reshape = op_arr.reshape((op_dim[0] * op_dim[1] * op_dim[2] * op_dim[3], op_dim[4]*op_dim[5]*op_dim[6]*op_dim[7]))
            rhs_dim = rhs.dimensions
            rhs_arr_reshape = rhs_arr.reshape((rhs_dim[0] * rhs_dim[1] * rhs_dim[2] * rhs_dim[3]))
            sol_arr = np.linalg.solve(op_arr_reshape,rhs_arr_reshape)

            sol_arr_reshape = sol_arr.reshape((op.dimensions[0] , op.dimensions[1] , op.dimensions[2],op.dimensions[3]))
            sol = xerus.Tensor.from_ndarray(sol_arr_reshape)
            C1.set_component(pos,sol)

            Ax = xerus.Tensor()
            Ax(i2,i1) << op_pre(j1,j2,j3,j4,i1,i2) * sol(j1,j2,j3,j4)
            error = (Ax-Y).frob_norm() / (Y.frob_norm())
            error2 = C1.frob_norm() 


            if forward and pos < noo - 1:
                C1.move_core(pos+1,True)
                C1_tmp = C1.get_component(pos)
                rStack = rStack[:-1]
                C1_tmp(i1,i2,i3,i4) << C1_tmp(i1,k1,k2,i4) * Ai(k1,i2) *C2i(k2,i3)
                lstacknp = lStack[-1].to_ndarray()
                C1_tmpnp = C1_tmp.to_ndarray()
                lstacknpres = np.einsum('dim,imdj->djm',lstacknp,C1_tmpnp)  
                lStack.append(xerus.Tensor.from_ndarray(lstacknpres))
            if not forward and pos > 0:
                C1.move_core(pos-1,True) 
                C1_tmp = C1.get_component(pos)
                lStack = lStack[:-1]
                C1_tmp(i1,i2,i3,i4) << C1_tmp(i1,k1,k2,i4) * Ai(k1,i2) *C2i(k2,i3)
                rstacknp = rStack[-1].to_ndarray()
                C1_tmpnp = C1_tmp.to_ndarray()
                rstacknpres = np.einsum('imdj,djm->dim',C1_tmpnp,rstacknp)  
                rStack.append(xerus.Tensor.from_ndarray(rstacknpres))  
            
            mem = pos
        #end of iteration
        #lam = lam/10
        lam =np.max([np.min([0.1*error/C1.frob_norm(),lam/4]),1e-14])
        diff = C1ex-C1
        tmp1 = xerus.Tensor()
        l = np.ones([noo,1,1,noo])
        r = np.ones([noo,1,1,noo])
        for i in range(noo):
            t1 = diff.get_component(i)
            t2 = C2list[i]
            tmp1(i1,i2,i3,j1,j2,j3) <<  t2(k1,i3) * t1(i1,k2,k1,j1) * t1(i2,k2,k3,j2)* t2(k3,j3)
            tmp1np = tmp1.to_ndarray()
            l = np.einsum('dije,ijdkle->dkle',l,tmp1np) 
        lr = np.sqrt(np.einsum('ijkl,ijkl->',l,r)) / C1ex.frob_norm()
        print("Iteration " + str(it) + ' Error: ' + str(lr) + " Residual: " + str(error) + " Norm: " + str(C1.frob_norm())+ " Lambda: " + str(lam))
        
        errors.append(lr)      
        errors2.append(error)
        lams.append(lam)
        
        
    return errors


def adapt_ranks(U, S, Vt,smin):
    """ Add a new rank to S
    Parameters
    ----------
    U: xerus Tensor
        left part of SVD
    S: xerus Tensor
        middle part of SVD, diagonal matrix
    Vt: xerus Tensor
        right part of SVD
    smin: float
        Threshold for smalles singluar values
    
    Returns
    -------
    Unew: xerus Tensor
        left part of SVD with one rank increased
    Snew: xerus Tensor
        middle part of SVD, diagonal matrix with one rank increased
    Vtnew: xerus Tensor
        right part of SVD with one rank increased
    """
    i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3 = xerus.indices(13)
    res = xerus.Tensor()
    #S
    Snew = xerus.Tensor([S.dimensions[0]+1,S.dimensions[1]+1])
    Snew.offset_add(S, [0,0])
    Snew[S.dimensions[0],S.dimensions[1]] = 0.01 * smin
    
    #U
    onesU = xerus.Tensor.ones([U.dimensions[0],U.dimensions[1]])
    Unew = xerus.Tensor([U.dimensions[0],U.dimensions[1],U.dimensions[2]+1])
    Unew.offset_add(U, [0,0,0])
    res(i1,i2) << U(i1,i2,k1) * U(j1,j2,k1) * onesU(j1,j2)
    onesU = onesU - res
    res(i1,i2) << U(i1,i2,k1) * U(j1,j2,k1) * onesU(j1,j2)
    onesU = onesU - res
    onesU.reinterpret_dimensions([U.dimensions[0],U.dimensions[1],1])
    onesU/= onesU.frob_norm()
    Unew.offset_add(onesU, [0,0,U.dimensions[2]])
    
    #Vt
    onesVt = xerus.Tensor.ones([Vt.dimensions[1],Vt.dimensions[2]])
    Vtnew = xerus.Tensor([Vt.dimensions[0]+1,Vt.dimensions[1],Vt.dimensions[2]])
    Vtnew.offset_add(Vt, [0,0,0])
    res(i1,i2) << Vt(k1,i1,i2) * Vt(k1,j1,j2) * onesVt(j1,j2)
    onesVt = onesVt - res
    res(i1,i2) << Vt(k1,i1,i2) * Vt(k1,j1,j2) * onesVt(j1,j2)
    onesVt = onesVt - res
    onesVt.reinterpret_dimensions([1,Vt.dimensions[1],Vt.dimensions[2]])
    onesVt/= onesVt.frob_norm()
    Vtnew.offset_add(onesVt, [Vt.dimensions[0],0,0])
    
    
    return Unew, Snew, Vtnew


def update_components_salsa(G,noo,d, Alist, nos, Y,smin, w, kminor,adapt,mR,maxranks):
    """Perform one SALSA sweep
    Parameters
    ----------
    G: xerus TTTensor
        iterate tensor
    noo: int
        number of dimension
    d: int
        number of dimension plus 1
    Alist: list of xerus tensor
        dictionary tensor   
    nos: int
        number of samples
    Y: xerus Tensor
        right hand side
    smin: float
        SALSA parameter smin
    w: float
        SALSA parameter omega
    kminor: int
        SALSA parameter number of additional ranks used in each simulation  
    adapt: bool
        if adaption should be used if necessary
    mR: list of int
        list of maximal ranks overall
    maxranks: int
        maximal rank alowed
    
    Returns
    -------
    error: float
        Residuum of iterate
    """
    
    p = Alist[0].dimensions[0]
    Smu_left,Gamma, Smu_right, Theta, U_left, U_right, Vt_left, Vt_right = (xerus.Tensor() for i in range(8))
    i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3 = xerus.indices(13)
    tmp = xerus.Tensor()

     # building Stacks for operators   
    lStack = [xerus.Tensor.ones([1,nos])]
    rStack = [xerus.Tensor.ones([1,nos])]

    G_tmp = G.get_component(noo)
    tmp(i1,i2,i3) << G_tmp(i1,i3,k1) * rStack[-1](k1,i2)
    rStack.append(tmp)
    for ind in range(d-2,0,-1): 
        G_tmp = G.get_component(ind)
        A_tmp = Alist[ind]
        G_tmp(i1,i2,i3) << G_tmp(i1,k1,i3) * A_tmp(k1,i2) 
        rstacknp = rStack[-1].to_ndarray()
        G_tmpnp = G_tmp.to_ndarray()
        rstacknpres = np.einsum('jmk,kms->jms',G_tmpnp,rstacknp)  
        rStack.append(xerus.Tensor.from_ndarray(rstacknpres))
    
    #loop over each component from left to right
    for mu in range(0,d):
        # get singular values and orthogonalize wrt the next core mu
        if mu > 0:
            # get left and middle component
            Gmu_left = G.get_component(mu-1)
            Gmu_middle = G.get_component(mu)
            (U_left(i1,i2,k1), Smu_left(k1,k2), Vt_left(k2,i3)) << xerus.SVD(Gmu_left(i1,i2,i3))
            Gmu_middle(i1,i2,i3) << Vt_left(i1,k2) *Gmu_middle(k2,i2,i3)
            #for j in range(kmin):
            if G.ranks()[mu-1] < np.min([maxranks[mu-1],mR]) and adapt \
                and Smu_left[int(np.max([Smu_left.dimensions[0] - kminor,0])),int(np.max([int(Smu_left.dimensions[1] - kminor),0]))] > smin:
                U_left, Smu_left, Gmu_middle  = adapt_ranks(U_left, Smu_left, Gmu_middle,smin)
            sing = [Smu_left[i,i] for i in range(Smu_left.dimensions[0])]
            
            Gmu_middle(i1,i2,i3) << Smu_left(i1,k1)*Gmu_middle(k1,i2,i3)
            G.set_component(mu-1, U_left)
            G.set_component(mu, Gmu_middle)
            Gamma = xerus.Tensor(Smu_left.dimensions) # build cut-off sing value matrix Gamma
            for j in range(Smu_left.dimensions[0]):
                Gamma[j,j] = 1 / np.max([smin,Smu_left[j,j]])
        if mu < d - 1:
            # get middle and rightcomponent
            Gmu_middle = G.get_component(mu)
            Gmu_right = G.get_component(mu+1)
            (U_right(i1,i2,k1), Smu_right(k1,k2), Vt_right(k2,i3)) << xerus.SVD(Gmu_middle(i1,i2,i3))


            sing = [Smu_right[i,i] for i in range(Smu_right.dimensions[0])]
            Gmu_right(i1,i2,i3) << Vt_right(i1,k1) *Gmu_right(k1,i2,i3)
            #if mu == d-2 and G.ranks()[mu] < maxranks[mu] and adapt and Smu_right[Smu_right.dimensions[0] - kminor,Smu_right.dimensions[1] - kminor] > smin:
            #    U_right, Smu_right, Gmu_right  = adapt_ranks(U_right, Smu_right, Gmu_right,smin)
            Gmu_middle(i1,i2,i3) << U_right(i1,i2,k1) * Smu_right(k1,i3)
            G.set_component(mu, Gmu_middle)
            G.set_component(mu+1, Gmu_right)
            Theta = xerus.Tensor([Gmu_middle.dimensions[2],Gmu_middle.dimensions[2]]) # build cut-off sing value matrix Theta
            for j in range(Theta.dimensions[0]):
                if j >= Smu_right.dimensions[0]:
                    sing_val = 0
                else:
                    singval = Smu_right[j,j] 
                Theta[j,j] = 1 / np.max([smin,singval])
       
    
    
        #update Stacks
        if mu > 0:
            G_tmp = G.get_component(mu-1)
            A_tmp = Alist[mu-1]
            G_tmp(i1,i2,i3) << G_tmp(i1,k1,i3) * A_tmp(k1,i2) 
            lstacknp = lStack[-1].to_ndarray()
            G_tmpnp = G_tmp.to_ndarray()       
            lstacknpres = np.einsum('jm,jmk->km',lstacknp,G_tmpnp)
            lStack.append(xerus.Tensor.from_ndarray(lstacknpres))
            rStack = rStack[:-1]

        op = xerus.Tensor()
        op_pre = xerus.Tensor()
        op_reg = xerus.Tensor()
        rhs = xerus.Tensor()
        Gi = G.get_component(mu)
        if mu != d-1:
            Ai = Alist[mu]
            Ainp = Ai.to_ndarray()
            lStacknp = lStack[-1].to_ndarray()
            rStacknp = rStack[-1].to_ndarray()
            op_pre_np = np.einsum('im,jm,kms->ijkms',lStacknp,Ainp,rStacknp)        
            op_pre = xerus.Tensor.from_ndarray(op_pre_np)
            op(i1,i2,i3,j1,j2,j3) << op_pre(i1,i2,i3,k1,k2) * op_pre(j1,j2,j3,k1,k2)
            rhs(i1,i2,i3) <<  op_pre(i1,i2,i3,k1,k2) * Y(k2,k1)
        else:
            tmp_id =  xerus.Tensor.identity([noo,1,noo,1])
            tmp_ones = xerus.Tensor.ones([1])
            op(i1,i2,i3,j1,j2,j3) << lStack[-1](i1,k1) *lStack[-1](j1,k1) *tmp_id(i2,i3,j2,j3)
            rhs(i1,i2,i3) <<  lStack[-1](i1,k1) * Y(i2,k1) *tmp_ones(i3)


        
        if mu < d - 1:
            id_reg_p = xerus.Tensor.identity([p,p])
        else:
            id_reg_p = xerus.Tensor.identity([noo,noo])
        if mu > 0:
            id_reg_r = xerus.Tensor.identity([Gi.dimensions[2],Gi.dimensions[2]])
            op_reg(i1,i2,i3,j1,j2,j3) << Gamma(i1,k1) * Gamma(k1,j1) * id_reg_r(i3,j3)  * id_reg_p(i2,j2)
            op += w*w * op_reg
        if mu < d-1:
            id_reg_l = xerus.Tensor.identity([Gi.dimensions[0],Gi.dimensions[0]])
            op_reg(i1,i2,i3,j1,j2,j3) << Theta(i3,k1) * Theta(k1,j3) * id_reg_l(i1,j1)  * id_reg_p(i2,j2)
            op += w*w * op_reg
        #if mu > 0 and mu < d - 1:
        #    op_reg(i1,i2,i3,j1,j2,j3) << Theta(i3,k1) * Theta(k1,j3) * Gamma(i1,k1) * Gamma(k1,j1)  * id_reg_p(i2,j2)
        #    op += w*w *w*w* op_reg

       

        op_arr = op.to_ndarray()
        rhs_arr = rhs.to_ndarray()
        gi_arr = Gi.to_ndarray()
   
        op_dim = op.dimensions
        op_arr_reshape = op_arr.reshape((op_dim[0] * op_dim[1] * op_dim[2], op_dim[3]*op_dim[4]*op_dim[5]))
        rhs_dim = rhs.dimensions
        rhs_arr_reshape = rhs_arr.reshape((rhs_dim[0] * rhs_dim[1] * rhs_dim[2]))
        gi_dim = Gi.dimensions
        gi_arr_reshape = gi_arr.reshape((gi_dim[0] * gi_dim[1] * gi_dim[2]))        
        
        
        
        sol_arr = np.linalg.solve(op_arr_reshape,rhs_arr_reshape)

        
        sol_arr_reshape = sol_arr.reshape((gi_dim[0], gi_dim[1], gi_dim[2]))
        sol = xerus.Tensor.from_ndarray(sol_arr_reshape)
        G.set_component(mu,sol)
        

        
        if mu != d-1:
            Ax = xerus.Tensor()
            Ax(i2,i1) << op_pre(j1,j2,j3,i1,i2) * sol(j1,j2,j3)
            error = (Ax-Y).frob_norm() / Y.frob_norm()

            #print("mu=" + str(mu) +'\033[1m'+" e=" + str(error)  +'\033[0m'+ ' nG=' + str(G.frob_norm()) + ' sing ' + str(sing[-kmin]))
    return error


def run_salsa(X,noo,Alist,nos,Solution,Y,maxiter,maxranks, kmin):
    """Perform the SALSA simulation
    Parameters
    ----------
    X: xerus TTTensor
        iterate tensor
    noo: int
        number of dimension
    Alist: list of xerus tensor
        dictionary tensor   
    nos: int
        number of samples
    Solution: xerus TTTensor
        exact solution
    Y: xerus Tensor
        right hand side
    maxiter: int
        number of iterations
    maxranks: list of int
        list of maximal ranks overall by given TT format
    kmin: int
        SALSA parameter number of additional ranks used in each simulation  
    Returns
    -------
    err_list2: list of floats
        list of relative error after each sweep
    """
    smin = 0.2#/np.sqrt(nos)*np.sqrt(np.power(p,noo)*noo)
    omega = 1.0
    fmin = 0.2
    fomega = 1.05
    omega_list = []
    smin_list = []
    err_list = []
    err_list2 = []
    maxRank = 30

    for iterate in range(0,maxiter):
        print("---------- iterate = " + str(iterate) + " omega = " + str(omega) + " smin = " + str(smin) + " error " + str(err_list2[-1] if iterate > 0 else 1)) 
        print(X.ranks()) 
        if omega < smin:
            break
        X.move_core(0)
        res = update_components_salsa(X,noo,noo+1,Alist,nos,Y,smin,omega, kmin, (True if iterate > 1 else False),maxRank,maxranks)


        omega = np.min([omega/fomega,np.sqrt(res)])
        omega = np.max([omega,res])
        smin = np.min([0.2*omega, 0.2*res]) #/np.sqrt(noo)*np.sqrt(np.power(p,noo)*noo)
        if res < 1e-14:
            break
        X.round(np.max([0.01*smin,1e-14]))
  

        err_list.append(res)
        err_list2.append((X-Solution).frob_norm()/Solution.frob_norm())

    return err_list2


    

