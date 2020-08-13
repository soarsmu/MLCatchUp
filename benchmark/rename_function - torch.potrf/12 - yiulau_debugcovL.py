import torch,numpy,time
dim = 200
X = torch.randn(dim,dim)
H = X.mm(X.t()) + torch.eye(dim,dim)

H_inv = torch.inverse(H)

L = torch.potrf(H,upper=False)

recomposedH = L.mm(L.t())

diff_H  = ((H - recomposedH)*(H - recomposedH)).sum()

print("diff H {}".format(diff_H))

L_inv = torch.inverse(L)

recomposedH_inv = L_inv.t().mm(L_inv)

diff_inv_H  = ((recomposedH_inv-H_inv)*(recomposedH_inv-H_inv)).sum()

print("diff H_inv {}".format(diff_inv_H))


# now we first take inverse and then decompose

L_inv = torch.potrf(H_inv,upper=False)

recomposedH_inv2 = L_inv.mm(L_inv.t())

diff_inv_H2 = ((recomposedH_inv2-H_inv)*(recomposedH_inv2-H_inv)).sum()

print("diff H_inv 2 {}".format(diff_inv_H2))


num_samples = 5

multi_L_time = 0
multi_H_time = 0
for i in range(num_samples):
    p = torch.randn(dim)
    temp = time.time()
    o = H.mv(p)
    multi_H_time += time.time()-temp
    temp = time.time()
    o = L.mv(p)
    multi_L_time += time.time()-temp

print("H multi time {}".format(multi_H_time))
print("L multi time {}".format(multi_L_time))

exit()
mean_inverse_time = 0
mean_L_inverse_time = 0
mean_potrs_time = 0
mean_diff = 0


# test computing L_inv p
p = torch.randn(dim)
L_inv = torch.inverse(L)
correct = L_inv.mv(p)
recomposed_correct,_  = torch.trtrs(p.unsqueeze(1),L,upper=False)

#print(correct)
#print(recomposed_correct)
#exit()
for i in range(num_samples):
    p = torch.randn(dim)
    temp = time.time()
    correct = L_inv.mv(p)
    mean_L_inverse_time += time.time()-temp
    temp = time.time()
    temp = time.time()
    o = H_inv.mv(p)

    recomposed_correct,_  = torch.trtrs(p.unsqueeze(1),L,upper=False)
    recomposed_correct = recomposed_correct.squeeze()
    mean_potrs_time += time.time() - temp
    diff = ((correct - recomposed_correct) * (correct - recomposed_correct)).sum()
    mean_diff += diff
#print(correct)
#print(recomposed_correct)
mean_inverse_time *= 1/num_samples
mean_potrs_time *= 1/num_samples
mean_diff *= 1/num_samples
print("mean inverse time {}".format(mean_inverse_time))
print("mean potrs time {}".format(mean_potrs_time))
print("mean diff {}".format(mean_diff))

diff = ((correct-recomposed_correct)*(correct-recomposed_correct)).sum()
print("diff invert lower triangular {}".format(diff))
exit()

# is it faster to invert L

invertL_time = 0
invertH_time = 0

for i in range(num_samples):
    X = torch.randn(dim, dim)
    H = X.mm(X.t()) + torch.eye(dim, dim)
    L = torch.potrf(H, upper=False)

    temp =  time.time()
    torch.inverse(H)
    invertH_time +=  time.time()-temp
    temp = time.time()
    torch.inverse(L)
    invertL_time += time.time()-temp


print("invert H time {}".format(invertH_time))
print("invert L time {}".format(invertL_time))
# answer is no
exit()
# test covariances


store = torch.zeros(num_samples,dim)
for i in range(num_samples):
    store[i,:] = L.mv(torch.randn(dim))


store_np = store.numpy()

emp_cov = numpy.cov(store_np,rowvar=False)

true_cov = H.numpy()
diff_cov = ((true_cov - emp_cov)*(true_cov-emp_cov)).sum()

print(true_cov.diagonal())
print(emp_cov.diagonal())
print("diff_cov {}".format(diff_cov))




