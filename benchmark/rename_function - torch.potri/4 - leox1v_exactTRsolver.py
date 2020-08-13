import torch
import numpy as np



from scipy import linalg
def mitternachtsformel(a,b,c):
    sqrt_discriminant = np.sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper



def alg736(grad, H_diag,H_offdiag, tr_radius, exact_tol,accepted_flag,previous_lambda, verbose=0): #for banded matrices


    verbose=False

    d=len(grad)
    precision=1E-8
    H = np.zeros((2,d))
    H[1,:] = np.array(H_diag) #+previous_lambda  ### Right now, previous lambda is not used.
    H[0,1:] = np.array(H_offdiag)
    HPD=False

    ###STEP 1&2: If H is psd set lambda=0 otherwise set lambda=-lambda_1+
  

    try: # See if H is psd, if so, directly obtain small h
        h=linalg.solveh_banded(H,-grad)
        HPD=True
        my_lambda=0

    except:
        HPD=False
    if not HPD: #if H is not psd, do an EVD to get the smallest EV and solve for small h again.
        ew,ev=linalg.eig_banded(H,lower=False,select='i', select_range=(0,0))
        if ew>=0: #this should not happen
            my_lambda=0
        else:
            my_lambda=-(ew[0] - precision) 
        H[1,:] = np.array(H_diag)+my_lambda #add lambda
        h=linalg.solveh_banded(H,-grad)


    hn=linalg.norm(h)
    ### STEP3: 
    if hn <=tr_radius:
        ### STEP3a
        if my_lambda==0 or np.abs(hn-tr_radius)<precision:
            if verbose:
                print('interior solution in alg 736')

            return h,my_lambda
        else: #we're in the hardcase :/
            a=1
            b=2*np.dot(h,ev)
            c=hn**2-tr_radius**2
            sqrt_discriminant= np.sqrt(b**2-4*a*c)
            alpha_lower=(-b+sqrt_discriminant)/(2*a)
            alpha_upper=(-b-sqrt_discriminant)/(2*a)
            flag=np.random.randint(2)
            if verbose:
                print('hard case in alg 736')

            return h+(flag*alpha_lower+(1-flag)*alpha_upper)*ev[0],my_lambda
        
    while True:
        if np.abs(hn-tr_radius)<=exact_tol*tr_radius:
            if verbose:
                print('easy case in alg 736')
            return h,my_lambda

        w_hat=linalg.solveh_banded(H,-h)
        wn_sq=np.dot(-w_hat,h)
        my_lambda+= ((hn-tr_radius)/tr_radius)*(hn**2/wn_sq)
        H[1,:]=np.array(H_diag)+my_lambda
        h=linalg.solveh_banded(H,-grad)
        hn=linalg.norm(h)

def alg736_restarted(grad, H_diag,H_offdiag, tr_radius, exact_tol,accepted_flag,previous_lambda, verbose=0): #for banded matrices
    d=len(grad)
    precision=1E-9
    H = np.zeros((2,d))
    H[0,1:] = np.array(H_offdiag)
    HplusLambdaPD=False
    HPD=False

    if previous_lambda>0:
        ###STEP 0: Check if previous lambda is in L and if so, that h(lambda_previous) is outside the Trust region
        H[1,:] = np.array(H_diag) +previous_lambda
        try: # Check if H+previous_lambda is PD and if so, compute small h
            h=linalg.solveh_banded(H,-grad)  #Fact(1)
            ## in this version it might be slower than w/o restarting!
            hn=linalg.norm(h)
            HplusLambdaPD=True
            if hn >=tr_radius:
                skip1and2=True
                my_lambda=previous_lambda
            else:
                skip1and2=False
        except:
            skip1and2=False
    else:
        skip1and2=False
        
    if skip1and2==False:
        H[1,:] = np.array(H_diag) #undo the addition of the previous lambda
        
        ###STEP 1: If H is psd set lambda=0 otherwise set lambda=-lambda_1+
        if HplusLambdaPD: #there is a chance of H to be psd
            try:#see if H is psd. If so, directly obtain small h (STEP 2a) and skip the eigendecomposition.
                h=linalg.solveh_banded(H,-grad)      #Fact(2)
                HPD=True
            except: #HPD stays false
                HPD=False
            
        if not HPD: #if H is not pd, compute the smallest eigval
            ew,ev=linalg.eig_banded(H,lower=False,select='i', select_range=(0,0))
            if ew>=0: #this should never happen now because the "try" would have gone through.
                my_lambda=0
            else:
                my_lambda=-(ew[0] - precision) #what exactly is lambda_1+??
            ### STEP2b: Factorize regularized H
            H[1,:] = np.array(H_diag)+my_lambda
            h=linalg.solveh_banded(H,-grad)      #Fact(2)

        hn=linalg.norm(h)
    ### STEP3: 
    if hn <=tr_radius:
        ### STEP3a
        if my_lambda==0 or np.abs(hn-tr_radius)<precision:
            if verbose:
                print('interior solution in alg 736 restarted')
            return h,my_lambda
        else: #we're in the hardcase :/
            a=1
            b=2*np.dot(h,ev)
            c=hn**2-tr_radius**2
            sqrt_discriminant= np.sqrt(b**2-4*a*c)
            alpha_lower=(-b+sqrt_discriminant)/(2*a)
            alpha_upper=(-b-sqrt_discriminant)/(2*a) ## both should be minimizers. I'll just randomize
            flag=np.random.randint(2)
            if verbose:
                print('hard case in alg 736 restarted')
            #IPython.embed()

            return h+(flag*alpha_lower+(1-flag)*alpha_upper)*ev[0],my_lambda
        
    while True:          

        if np.abs(hn-tr_radius)<=exact_tol*tr_radius:
            if verbose:
                print('easy case in alg 736 restarted')
            return h,my_lambda

        w_hat=linalg.solveh_banded(H,-h) #Fact(3). h itself comes from Fact(1) or Fact(2).
        wn_sq=np.dot(-w_hat,h)
        my_lambda+= ((hn-tr_radius)/tr_radius)*(hn**2/wn_sq)
        H[1,:]=np.array(H_diag)+my_lambda
        h=linalg.solveh_banded(H,-grad)
        hn=linalg.norm(h)

 


def alg734_cpu(grad, T_diag,T_offdiag, tr_radius, exact_tol, accepted_flag,previous_lambda):
    from scipy import linalg
    from scipy.sparse import diags
    from math import sqrt
    d = len(T_diag)
    
    Hdiag=np.array(T_diag)
    Hoffdiag=np.array(T_offdiag)

    H_band = np.zeros((2,d))
    H_band[1,:] = Hdiag
    H_band[0,1:] = Hoffdiag
    s = np.zeros_like(grad)
    
    ## Step 0: initialize safeguards
    absHdiag = np.abs(Hdiag)
    absHoffdiag = np.abs(Hoffdiag)
    H_ii_min = min(Hdiag)
    H_max_norm = d * max(absHoffdiag.max(),absHdiag.max())
    #H_fro_norm = np.sqrt(2*np.dot(absHoffdiag,absHoffdiag) + np.dot(absHdiag,absHdiag)) #no crossterms here?
    H_fro_norm= np.linalg.norm(H_band, 'fro')


    gerschgorin_l = max([Hdiag[0]+absHoffdiag[0] , Hdiag[d-1]+absHoffdiag[d-2]])
    gerschgorin_l = max([gerschgorin_l]+[Hdiag[i]+absHoffdiag[i]+absHoffdiag[i-1] for i in range(1,d-1)])#see conn2000,sect 7.3.8, \lambda^L
    gerschgorin_u = max([-Hdiag[0]+absHoffdiag[0] , -Hdiag[d-1]+absHoffdiag[d-2]])
    gerschgorin_u = max([gerschgorin_u]+[-Hdiag[i]+absHoffdiag[i]+absHoffdiag[i-1] for i in range(1,d-1)])
    lambda_lower = max(0, -H_ii_min, np.linalg.norm(grad) / tr_radius - min(H_fro_norm, H_max_norm, gerschgorin_l))
    lambda_upper = max(0, np.linalg.norm(grad) / tr_radius + min(H_fro_norm, H_max_norm, gerschgorin_u))


#### Alternatively:
    #gerschgorin_l = max([H[i, i] + (np.sum(np.abs(H[i, :])) - np.abs(H[i, i])) for i in range(len(H))])
    #gerschgorin_u = max([-H[i, i] + (np.sum(np.abs(H[i, :])) - np.abs(H[i, i])) for i in range(len(H))])

    #print('diag',lambda_lower,lambda_upper,gerschgorin_l,gerschgorin_u,H_ii_min,H_max_norm,H_fro_norm)

    if accepted_flag==False and lambda_lower <= previous_lambda <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
        lambda_j=previous_lambda
    elif lambda_lower == 0:  # allow for fast convergence in case of inner solution
        lambda_j = lambda_lower
    else:
        #lambda_j = (lambda_upper+lambda_lower)/2
        lambda_j=np.random.uniform(lambda_lower, lambda_upper)

    i=0
    # Root Finding
    phi_history = []
    lambda_history = []

    use_L=False
    while True: ## We are looking for a root of phi(lambda)=1/||s(lambda)||-1/tr_radius, where s(lambda)= -(B+lambda*I)^{-1}g (7.2.7)
        i+=1
        lambda_in_N = False
        lambda_plus_in_N = False
        #B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
        B = np.zeros((2,d))
        B[1,:] = Hdiag + lambda_j
        B[0,1:] = Hoffdiag
        try:     ## If this succeeds then Lambda is in F (feasible)
  
            #2. Compute L and solve banded again
          
            # 1 Factorize B (for w later)

            if use_L:
                L = linalg.cholesky_banded(B) #comes back in compact form
                LT[0,:]=L[1,:]
                LT[1,0:-1]=L[0,1:]
                LT[1,-1]=0
                # 2 Solve LL^Ts=-g
            s = linalg.solveh_banded(B,-grad)

            sn = np.linalg.norm(s)
            
            ## 2.1 Termination: Lambda in F, if q(s(lamda))<eps_opt q(s*) and sn<eps_tr tr_radius -> stop. By Conn: Lemma 7.3.5:
            phi_lambda = 1. / sn - 1. / tr_radius
            
            #if (abs(sn - tr_radius) <= exact_tol * tr_radius):
            if i>1 and phi_lambda in phi_history: #detect if we are caught in a loop due to finite precision
                lambda_j = lambda_history[np.argmin(phi_history)] #pick best lambda observed so far and exit
                break
            else:
                phi_history.append(phi_lambda)
                lambda_history.append(lambda_j)
            if (abs(phi_lambda)<= exact_tol): #
                break

            # 3 Solve Lw=s  <- we are aiming for phi' (lambda) here and need nabla_lambda s(lambda) (Eq 7.39) or directly the dot product of this with s, which can also be computed as (7.3.11.1/2) 
            # but that latter needs L and since we don't explicitly compute L to factorize the banded version of B above it's extra work. I thus suggest to go with use_L=False.
            if use_L:
                w=linalg.solve_banded((1,0),LT,s)
                wn = np.linalg.norm(w)
            else:
                w_hat=linalg.solveh_banded(B,-s)
                wn=np.sqrt(np.dot(-w_hat,s))


            ##Step 1: Lambda in L. 
            if lambda_j > 0 and (phi_lambda) < 0:
                #print ('lambda: ',lambda_j, ' in L')
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)
                lambda_j = lambda_plus


            ##Step 2: Lambda in G    (sn<tr_radius)
            elif (phi_lambda) > 0 and lambda_j > 0: #TBD: remove grad
                lambda_upper = lambda_j
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)

                ##Step 2a: If factorization succeeds: lambda_plus in L
                if lambda_plus > 0:
                    try:
                        # 1 Factorize B
                        B_plus = np.zeros((2,d))
                        B_plus[1,:] = Hdiag + lambda_plus
                        B_plus[0,1:] = Hoffdiag
                        _ = linalg.cholesky_banded(B_plus) # throws LinAlgError of B_plus is not pos.def.
                        lambda_j = lambda_plus

                    except np.linalg.LinAlgError:
                        lambda_plus_in_N = True

                ##Step 2b/c: If not: Lambda_plus in N
                if lambda_plus <= 0 or lambda_plus_in_N == True:
                    # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
                    try:
                        _ = linalg.cholesky_banded(H_band)     # B???
                        H_pd = True
                    except np.linalg.LinAlgError:
                        H_pd = False

                    if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: #cannot happen in ARC!
                        lambda_j = 0
                        break
                        
                    # 2. Else, choose a lambda within the safeguard interval
                    else:
                        lambda_lower = max(lambda_lower, lambda_plus)  # reset lower safeguard
                        lambda_j = max(sqrt(lambda_lower * lambda_upper),
                                       lambda_lower + 0.01 * (lambda_upper - lambda_lower))

                        lambda_upper = float(
                            lambda_upper) 
                        #if lambda_lower == lambda_upper:
                        if lambda_upper <= np.nextafter(lambda_lower,lambda_upper,dtype=Hdiag.dtype):
                            lambda_j = lambda_lower


                            ew,ev=linalg.eig_banded(H_band,lower=False,select='i', select_range=(0,0))  
                            


 
                            tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,ev), np.dot(s,s)-tr_radius**2) #Am I sure that this is correct??
                            s=s + tao_lower * ev #why tau lower?
                            print ('hard case resolved inside')
                            
                            return s,lambda_j

            elif (phi_lambda) == 0: ## interior converence??
                break
            else:      #TBD:  move into if lambda+ column #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                lambda_in_N = True
        ##Step 3: Lambda in N
        except np.linalg.LinAlgError:
            lambda_in_N = True
        if lambda_in_N == True:
            try:
                _ = linalg.cholesky_banded(H_band)  
                H_pd = True
            except np.linalg.LinAlgError:
                H_pd = False

            # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0) ### why do we do this here? and in first iteration phi_lambda is unknown
            if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: 
                lambda_j = 0
                break
            # 2. Else, choose a lambda within the safeguard interval
            else:
                lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
                lambda_j = max(sqrt(lambda_lower * lambda_upper),
                               lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.14
                lambda_upper = float(lambda_upper)  
                
                # Check for Hard Case:
                #print('884',i,lambda_upper-lambda_lower,np.nextafter(lambda_lower,lambda_upper)-lambda_lower,np.nextafter(1,2,dtype=Hdiag.dtype)-1,np.nextafter(1,2,dtype=grad.dtype)-1)
                if lambda_upper <= np.nextafter(lambda_lower,lambda_upper,dtype=Hdiag.dtype): #?????
                #if lambda_lower == lambda_upper:
                    lambda_j = lambda_lower
                    ew,ev = linalg.eig_banded(H_band,select='i',select_range=(0,0))  
                    ev=np.squeeze(ev)
                    
                    #assert usually doesn't hold in finite precision (only approximately)
                    #assert (ew == -lambda_j), "Ackward: in hard case but lambda_j != -lambda_1"
                    #IPython.embed()
                    tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,ev), np.dot(s,s)-tr_radius**2) #Am I sure that this is correct??
                    s=s + tao_lower * ev

                    print ('hard case resolved outside')
                
                    return s,lambda_j



    # compute final step
    B = np.empty((2,d))
    B[1,:] = Hdiag + lambda_j
    B[0,1:] = Hoffdiag
    # Solve LL^Ts=-g
    s = linalg.solveh_banded(B,-grad)
    return s,lambda_j



def alg734(grad, H, tr_radius, exact_tol,successful_flag,lambda_k):

    dtype=grad.type()
    s = torch.zeros_like(grad).type(dtype)
    precision=1E-30
    
    ## Step 0: initialize safeguards
    H_ii_min = min(torch.diagonal(H))
    H_max_norm = H.shape[0]* torch.abs(H).max()
    H_fro_norm = torch.norm(H)
    gerschgorin_l = max([H[i, i] + (torch.sum(torch.abs(H[i, :])) - torch.abs(H[i, i])) for i in range(len(H))])
    gerschgorin_u = max([-H[i, i] + (torch.sum(torch.abs(H[i, :])) - torch.abs(H[i, i])) for i in range(len(H))])
    #### this can be optimized for tridiagonal matrices

    lambda_lower = max(torch.zeros(1).squeeze().type(dtype), -H_ii_min, torch.norm(grad) / tr_radius - min(H_fro_norm, H_max_norm, gerschgorin_l))
    lambda_upper = max(torch.zeros(1).squeeze().type(dtype), torch.norm(grad) / tr_radius + min(H_fro_norm, H_max_norm, gerschgorin_u))

    if successful_flag==False and lambda_lower <= lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
        lambda_j=lambda_k
    elif lambda_lower == 0:  # allow for fast convergence in case of inner solution
        lambda_j = lambda_lower
    else:
        lambda_j=(lambda_upper-lambda_lower)*torch.rand(1).type(dtype)+lambda_lower

    i=0
    # Root Finding
    while True:
        i+=1
        lambda_in_N = False
        lambda_plus_in_N = False
        B = H + lambda_j * torch.eye(H.shape[0], H.shape[1]).type(dtype)
        try:
            # 1 Factorize B            
            L=torch.potrf(B,upper=False)
            
            # 2 Solve LL^Ts=-g
            
            LiLiTinv=torch.potri(L,upper=False)    #returns (LL^T)^{-1}
            s=-torch.mv(LiLiTinv,grad)
            #s=torch.potrs(-grad,L,upper=False)
            sn = torch.norm(s)
            
    
            
            ## 2.1 Termination: Lambda in F, if q(s(lamda))<eps_opt q(s*) and sn<eps_tr tr_radius -> stop. By Conn: Lemma 7.3.5:
            phi_lambda = 1. / sn - 1. / tr_radius
            #if (abs(sn - tr_radius) <= exact_tol * tr_radius):
            if (abs(phi_lambda)<=exact_tol): #
                break;

            # 3 Solve Lw=s
            w = torch.mv(torch.inverse(L),s)   ### I guess L^{-1} is computed in torch.potri but I don't know how to reconstruct it.
            wn = torch.norm(w)

            
            ##Step 1: Lambda in L
            if lambda_j > 0 and (phi_lambda) < 0:
                # print ('lambda: ',lambda_j, ' in L')
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)
                lambda_j = lambda_plus


            ##Step 2: Lambda in G    (sn<tr_radius)
            elif (phi_lambda) > 0 and lambda_j > 0 and torch.norm(grad) != 0: #TBD: remove grad
                # print ('lambda: ',lambda_j, ' in G')
                lambda_upper = lambda_j
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)

                ##Step 2a: If factorization succeeds: lambda_plus in L
                if lambda_plus > 0:
                    try:
                        # 1 Factorize B
                        B_plus = H + lambda_plus * torch.eye(H.shape[0], H.shape[1])
                        L = torch.potrf(B_plus,upper=False)
                        lambda_j = lambda_plus
                        # print ('lambda+', lambda_plus, 'in L')


                    except RuntimeError:
                        lambda_plus_in_N = True

                ##Step 2b/c: If not: Lambda_plus in N
                if lambda_plus <= 0 or lambda_plus_in_N == True:
                    # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)   <---- why do we check this here. also phi(lambda) is not known here (at least in iter 0)
                    try:
                        U = torch.potrf(H,upper=False)
                        H_pd = True
                    except RuntimeError:
                        H_pd = False

                    if torch.abs(lambda_lower) <precision and H_pd == True and phi_lambda >= 0: #cannot happen in ARC!
                        lambda_j = 0
                        #print ('inner solution found')
                        break
                    # 2. Else, choose a lambda within the safeguard interval
                    else:
                        # print ('lambda_plus', lambda_plus, 'in N')
                        lambda_lower = max(lambda_lower, lambda_plus)  # reset lower safeguard
                        lambda_j = max(torch.sqrt(lambda_lower * lambda_upper),
                                       lambda_lower + 0.01 * (lambda_upper - lambda_lower))

                         #lambda_upper = np.float32(  lambda_upper) 
                        
                        if torch.abs(lambda_lower - lambda_upper) < precision:
                            lambda_j = lambda_lower
                            ## Hard case
                            eigdec=torch.symeig(H,eigenvectors=True) #returns evals and evecs in ascending order ... maybe a partial eigdec would be faster but I don't know where to find that
                            #are we sure that this is not B???

                            ew=eigdec[0][0]
                            ev=eigdec[1][0]
                            #ew, ev = linalg.eigh(H, eigvals=(0, 0)) #returns the smallest EVec and EVal
                            assert (torch.abs(ew == -lambda_j)<precision), "Ackward: in hard case but lambda_j != -lambda_1"
                            #import IPython; IPython.embed(); exit(1)
                            
                            a=1#torch.mm(ev.t(),ev)
                            b=2*torch.mv(ev,s)
                            c=sn**2 -tr_radius**2
                            sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
                            tao_lower=(-b + sqrt_discriminant) / (2 * a)
                            s=s + tao_lower * ev  
                            print ('hard case resolved inside')

                            return s

            elif torch.abs(phi_lambda)<precision: 
                break
            else:
                lambda_in_N = True
        ##Step 3: Lambda in N
        except RuntimeError:
            lambda_in_N = True
        if lambda_in_N == True:
            # print ('lambda: ',lambda_j, ' in N')
            try:
                U = torch.potrf(H,upper=False)
                H_pd = True
            except RuntimeError:
                H_pd = False

            # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
            if torch.abs(lambda_lower)<precision and H_pd == True and phi_lambda >= 0: 
                lambda_j = 0
                #print ('inner solution found')
                break
            # 2. Else, choose a lambda within the safeguard interval
            else:
                lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
                lambda_j = max(torch.sqrt(lambda_lower * lambda_upper),
                               lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.14
                #lambda_upper = np.float32(lambda_upper)  
                # Check for Hard Case:
                if torch.abs(lambda_lower == lambda_upper)<precision:
                    lambda_j = lambda_lower
                    eigdec=torch.symeig(H,eigenvectors=True) #returns evals and evecs in ascending order ... maybe a partial eigdec would be faster but I don't know where to find that
                    ew=eigdec[0][0]
                    ev=eigdec[1][0]                  
                    assert (torch.abs(ew == -lambda_j)<precision), "Ackward: in hard case but lambda_j != -lambda_1"
                    #import IPython; IPython.embed(); exit(1)

                    a=1#torch.mm(ev.t(),ev)
                    b=2*torch.mv(ev,s)
                    c=sn**2 -tr_radius**2
                    sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
                    tao_lower=(-b + sqrt_discriminant) / (2 * a)
                    s=s + tao_lower * ev  
                    print ('hard case resolved outside')
                    return s




    # compute final step
    B = H + lambda_j * torch.eye(H.shape[0], H.shape[1]).type(dtype)
    
    # 1 Factorize B            
    L=torch.potrf(B,upper=False)
            
    # 2 Solve LL^Ts=-g
            
    LiLiTinv=torch.potri(L,upper=False)    #returns (LL^T)^{-1}
    s=-torch.mv(LiLiTinv,grad)
    #s=torch.potrs(-grad,L,upper=False)            

    return (s,lambda_j)
