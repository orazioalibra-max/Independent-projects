import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.tri as mtri

np.random.seed(42)

nx=10
ny=nx
xs=np.linspace(0,1,nx+1) 
ys=np.linspace(0,1,ny+1)

target_iter=50
alphas=10.0**np.array([-1,-2,-3,-4,-5,-6]) #regularization weight
tau_alphas=1.0/alphas #gradient descent step size, different for each alpha, since the gradient is proportional to  alpha

def ud_sinusoid(x,y): #target state, arbitrary function
    return np.sin(np.pi*x)*np.sin(np.pi*y)

def create_trimesh(nx,ny,xs,ys): #create a triangular mesh

    nodes=[]
    for x in xs:        
        for y in ys:     
            nodes.append([x, y])
    nodes=np.array(nodes) # shape ((nx+1)*(ny+1), 2)

    def idx(i, j): #returns the index of the node wrt nodes array
        return i*(ny+1)+j

    elements=[]
    for i in range(nx):
        for j in range(ny):
            a=idx(i, j)
            b=idx(i+1, j)
            c=idx(i+1, j+1)
            d=idx(i, j+1)
            elements.append([a, b, c])
            elements.append([a, c, d]) #shape (nx*ny*2,3) 
    return nodes, np.array(elements)

# compute area triangles
def area_triangles(tri): # tri=[x1,y1;x2,y2;x3,y3]
    tri=np.transpose(tri)
    return abs(((tri[0,1]*tri[1,2]-tri[1,1]*tri[0,2])-(tri[0,0]*tri[1,2]-tri[1,0]*tri[0,2])+(tri[0,0]*tri[1,1]-tri[1,0]*tri[0,1]))/2)

def dof_definition(nodes):
    n_nodes=len(nodes)

    dof=-np.ones(n_nodes, dtype=int)
    n_dof=0
    for i in range(n_nodes):
        x, y = nodes[i]
        on_boundary = (x == 0 or x == 1 or y == 0 or y == 1)
        if not on_boundary:
            dof[i]=n_dof
            n_dof+=1
    return dof, n_dof

def dirichlet(u,dof):

    n_full=np.shape(dof)[0]
    u_new=np.zeros(n_full)
    
    for i in range(n_full):
        if dof[i]==-1:
            u_new[i]=0.0 #useful when !=0
        else:
            u_new[i]=u[dof[i]]

    return u_new

def local_stiffness(tri,area):

    #change of variable from reference triangle to the actual triangle
    b=np.zeros((3,2))

    b[0,0]=tri[1,1]-tri[2,1]
    b[1,0]=tri[2,1]-tri[0,1]
    b[2,0]=tri[0,1]-tri[1,1]
    
    b[0,1]=tri[2,0]-tri[1,0]
    b[1,1]=tri[0,0]-tri[2,0]
    b[2,1]=tri[1,0]-tri[0,0]

    mat=np.zeros((3,3))
    for i in range(3):
        for j in range(i,3):
            mat[i,j]=b[i,0]*b[j,0]+b[i,1]*b[j,1]
            mat[j,i]=mat[i,j]
    mat=mat/(4*area)
    return mat

def local_mass(area):

    mat=np.zeros((3,3))
    for i in range(3):
        for j in range(i,3):
            if i==j:
                mat[i,j]=area/6
            else:
                mat[i,j]=area/12
                mat[j,i]=mat[i,j]
    return mat

nodes, elements = create_trimesh(nx, ny, xs, ys)
dof, n_dof = dof_definition(nodes)

#Build global matrices A and M
A = sp.lil_matrix((n_dof, n_dof))
M = sp.lil_matrix((n_dof, n_dof))

for elem in range(len(elements)):
    idxs=elements[elem,:]
    # tri=np.zeros((3,2))
    # r_dof=np.zeros(3)

    # for i, idx in enumerate(idxs):
    #     tri[i]=nodes[idx]
    #     r_dof[i]=dof[idx]
    tri=nodes[idxs] 
    r_dof=dof[idxs]

    area=area_triangles(tri)
    A_elem=local_stiffness(tri,area)
    M_elem=local_mass(area)

    for i in range(3):
        for j in range(3):
            if r_dof[i] !=-1 and r_dof[j]!= -1:
                A[r_dof[i],r_dof[j]]+=A_elem[i,j]
                M[r_dof[i],r_dof[j]]+=M_elem[i,j]

A=A.tocsr()
M=M.tocsr()

u_d=np.zeros(n_dof)
for idx, d in enumerate(dof):
    if d!=-1:
        x,y=nodes[idx]
        u_d[d]=ud_sinusoid(x,y)

def optimization_cycle(alpha, tau, u_d, max_iter, tol, print_log=False):
    z=np.ones(n_dof)

    for iter in range(max_iter):

        u=spla.spsolve(A,M@z)
        p=spla.spsolve(A,M@(u-u_d))
        g=alpha*z+p
        norm_g = np.linalg.norm(g)
                
        if print_log and iter%100 == 0:
            print(f" iter={iter} |g|={norm_g:.4e}")

        if (norm_g<tol):
            break

        z=z-tau*g
    return u, z, iter, np.linalg.norm(g)

def gradient_check(alpha,u_d):
    z_test=np.random.randn(n_dof)
    u=spla.spsolve(A,M@z_test)
    p=spla.spsolve(A,M@(u-u_d))
    g_test=M@(alpha*z_test+p)

    #finite differences
    eps=1e-6
    k=7  #index to perturbate, change to check different one
    def cost(z):
        u=spla.spsolve(A,M@z)
        r=u-u_d
        return 0.5*r@(M@r) + alpha*0.5*z@(M@z)

    z_plus=z_test.copy() 
    z_plus[k]+=eps

    z_minus=z_test.copy()
    z_minus[k]-=eps

    g_fd_k=(cost(z_plus)-cost(z_minus))/(2*eps) #numerical approximation of the gradient

    err=abs(g_test[k]-g_fd_k)/abs(g_test[k])
    print("Relative error in gradient:", err)
    return err

def recoverability_test(alphas):
    z_true = spla.spsolve(M, A@u_d)   # z is usually unknown, here we compute it with the true u_d so it is correct
    u_d_true=u_d
    err=np.zeros(len(alphas))

    for i in range(len(alphas)): #we try to find z_true trough z_alpha
        tau=min(150, tau_alphas[i])
        max_iter=int(target_iter/(tau*alphas[i]))
        tol=1e-4*(alphas[i]**2)

        _, z_alpha, iter, norm_g=optimization_cycle(alphas[i], tau, u_d_true, max_iter, tol, print_log=True)
        err[i]=np.linalg.norm(z_alpha-z_true)/np.linalg.norm(z_true)
        print("Recoverability for alpha={:.0e}: ".format(alphas[i]), f"err={err[i]:.4e}", " iterations: ", iter, " ||g|| = ", norm_g)

    plt.loglog(alphas,err,'*',label='error behaviour')
    plt.loglog(alphas,err[-1]*(alphas/alphas[-1])**(1),label='slope 1')
    plt.xlabel('alpha values')
    plt.ylabel('relative error')
    plt.title(f"Error behaviour with respect to alpha")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()
    return

print("gradient check: ", gradient_check(alphas[0], u_d))

#compute and plot solution (smallest alpha)
alpha=alphas[-1]
tau=min(150, tau_alphas[-1])
max_iter_sol=int(target_iter/(tau*alpha))
tol=1e-4*(alpha**2)

u, z, iter, norm_g=optimization_cycle(alpha, tau, u_d, max_iter_sol, tol)
print(f"||z||={np.linalg.norm(z):.4f}  ||u-u_d||={np.linalg.norm(u-u_d):.4e}  iter={iter}  ||g||={norm_g:.4e}")

u_sol=dirichlet(u,dof)
z_sol=dirichlet(z,dof)
u_d_full=dirichlet(u_d,dof)

triangulation=mtri.Triangulation(nodes[:,0], nodes[:,1], elements) #create triangulation (default: Delaunay)

#create 3 plots with u_d, u and z
fig, axs =plt.subplots(1,3,figsize=(16,4))

plot_ud=axs[0].tricontourf(triangulation, u_d_full, levels=50, cmap='viridis')
axs[0].set_title("Target state $u_d$")
fig.colorbar(plot_ud,ax=axs[0])

plot_u=axs[1].tricontourf(triangulation, u_sol, levels=50, cmap='viridis')
axs[1].set_title("Achieved state $u$")
fig.colorbar(plot_u,ax=axs[1])

plot_z=axs[2].tricontourf(triangulation, z_sol, levels=50, cmap='plasma')
axs[2].set_title("Optimal control $z$")
fig.colorbar(plot_z,ax=axs[2])

plt.tight_layout()
plt.show()

#check error decay rate
recoverability_test(alphas)