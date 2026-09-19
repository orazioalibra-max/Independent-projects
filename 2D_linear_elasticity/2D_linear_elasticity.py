
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

################################################################################
########                2D LINEAR ELASTICITY BVP (FEM P1)               ########
########                                                                ########
########  ⎧ -∇ · σ(u) = 0                                in Ω           ########
########  ⎪  σ(u) = ℂ : ε(u),  ε(u) = ½(∇u + ∇uᵀ)       in Ω           ########
########  ⎨  u = 0                                       on Γ_D         ########
########  ⎪  σ(u) · n = t,     t = [0, t_y(y)]ᵀ          on Γ_N         ########
########  ⎩  σ(u) · n = 0                                on Γ_free      ########
########                                                                ########
########  Domain:                                                       ########
########    Ω = (0, L) × (-h/2, h/2)                                    ########
########                                                                ########
########  Boundary Partition (∂Ω = Γ_D ∪ Γ_N ∪ Γ_free, disjoint):       ########
########    • Γ_D    = { x = 0 }      (Dirichlet: clamped support)      ########
########    • Γ_N    = { x = L }      (Neumann: applied shear traction) ########
########    • Γ_free = { y = ±h/2 }   (Free traction surfaces)          ########
################################################################################

L=8 #length of the beam
h=1 #height of the beam
E=1 #simplified Young's modulus
nu=0.3 #Poisson's ratio
P=1 #load at the end of the beam
ny=20
nx=ny*8

I=h**3/12 #moment of inertia for rectangulus
G=E/(2*(1+nu)) #shear modulus (tangent deformation)
As=(5/6)*h #unitary thickness of the beam (rectangular cross-section)

C=(E/((1+nu)*(1-2*nu)))*np.array([[1-nu,nu,0],
                                    [nu,1-nu,0],
                                    [0,0,(1-2*nu)/2]])

E_bend = E/(1-nu**2)  #effective bending modulus under plane strain 
delta_ex = P*L**3/(3*E_bend*I) + (P*L)/(G*As)  #Timoshenko's external deformation, corrected for plane strain

def create_trimesh(L,h,nx,ny): #create a triangular mesh
    xs=np.linspace(0,L,nx+1)
    ys=np.linspace(-h/2,h/2,ny+1) #nx and ny better be in coherence with 8:1 ratio

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
def area_triangles(p): # p=[x1,y1;x2,y2;x3,y3]
    p=np.transpose(p)
    return abs(((p[0,1]*p[1,2]-p[1,1]*p[0,2])-(p[0,0]*p[1,2]-p[1,0]*p[0,2])+(p[0,0]*p[1,1]-p[1,0]*p[0,1]))/2)

def strain_displacement_matrix(p, area):

    #change of variable from reference triangle to the actual triangle
    b=np.zeros((3,2))    

    b[0,0]=p[1,1]-p[2,1]
    b[1,0]=p[2,1]-p[0,1]
    b[2,0]=p[0,1]-p[1,1]
    
    b[0,1]=p[2,0]-p[1,0]
    b[1,1]=p[0,0]-p[2,0]
    b[2,1]=p[1,0]-p[0,0]

    #build B matrix

    B=(1/(2*area))*np.array([[b[0,0],0,b[1,0],0,b[2,0],0],
                             [0,b[0,1],0,b[1,1],0,b[2,1]],
                             [b[0,1],b[0,0],b[1,1],b[1,0],b[2,1],b[2,0]]])
    return B

def stiffness_matrix(B, C, area):
    return np.transpose(B)@C@B*area

def dof_definition(nodes,ny):

    nb_nodes=np.shape(nodes)[0]
    dof=-np.ones(2*nb_nodes, dtype=int)
    n_dof=0
    for i in range(2*(ny+1),2*nb_nodes):
            dof[i]=n_dof
            n_dof+=1
    return dof, n_dof

def dirichlet(u,dof):

    n_full=np.shape(dof)[0]
    u_new=np.zeros(n_full)
    
    for i in range(n_full):
        if dof[i]==-1:
            u_new[i]=0.0
        else:
            u_new[i]=u[dof[i]]

    return u_new

def second_member(nodes, nx, ny, dof, n_dof):
    F=np.zeros(n_dof)

    #ADD THE SECOND MEMBER HERE (in this case is 0)
    
    #Gauss formula with n=2 points in [-1,1] (2n-1=3)
    pt1=-1/np.sqrt(3)
    pt2=1/np.sqrt(3)
    w1=1
    w2=1

    def ty(y): # trasverse shear stress distribution (only on x==L)
        return (P/(2*I))*(h**2/4-y**2) 

    for j in range (ny):

        node1 = nx*(ny+1) + j
        node2 = nx*(ny+1) + j+1

        ynode1= nodes[node1, 1]
        ynode2= nodes[node2, 1]

        length=ynode2-ynode1

        F_local_1=0
        F_local_2=0

        for pt,w in [(pt1,w1),(pt2,w2)]:

            ynew=ynode1+(pt+1)/2*length #from [-1,1] to [ynode1,ynode2]

            phi1=(1-pt)/2 #value phi on the edge
            phi2=(1+pt)/2

            t=ty(ynew) #evaluate

            F_local_1+=w*t*phi1*(length/2)
            F_local_2+=w*t*phi2*(length/2)

        dof_y1=dof[2*node1+1]
        dof_y2=dof[2*node2+1]
        F[dof_y1]+=F_local_1
        F[dof_y2]+=F_local_2

    return F

def solve_beam(nx,ny):

    nodes, elements = create_trimesh(L, h, nx, ny)
    dof, n_dof = dof_definition(nodes, ny)

    #Build global stiffness matrix and rhs
    A = sp.lil_matrix((n_dof, n_dof))

    for elem in range(len(elements)):
        idxs=elements[elem,:]
        pt_triangle=np.zeros((3,2))
        global_dofs=np.zeros(6, dtype=int) # otherwise float, but indexes must be int
        reduced_dofs=np.zeros(6, dtype=int)

        for i in range(3):
            pt_triangle[i]=nodes[idxs[i],:]
            global_dofs[2*i]=2*idxs[i] #global u_x and u_y indexes in nodes
            global_dofs[(2*i)+1]=(2*idxs[i])+1

            reduced_dofs[2*i]=dof[global_dofs[2*i]]
            reduced_dofs[(2*i)+1]=dof[global_dofs[(2*i)+1]]

        area=area_triangles(pt_triangle)
        B_elem=strain_displacement_matrix(pt_triangle,area)
        A_elem=stiffness_matrix(B_elem,C,area)
        for i in range(6):
            for j in range(6):
                if reduced_dofs[i]!=-1 and reduced_dofs[j]!=-1: #filter dof from dirichlet nodes
                    A[reduced_dofs[i], reduced_dofs[j]]+=A_elem[i,j]

    F = second_member(nodes, nx, ny, dof, n_dof)
    A = A.tocsr()
    u = spla.spsolve(A, F)

    #apply Dirichlet condition
    u_approx=dirichlet(u,dof) #shape (2*(nx+1)*(ny+1), 1)

    beam_idx=nx*(ny+1)+ny//2 #node in nodes at (L,0)

    beam=u_approx[(2*beam_idx)+1] # u_y at (L,0)

    return beam, u_approx, nodes, elements

beam, u_approx, nodes, elements=solve_beam(nx,ny)

# err=abs(delta_ex-beam)

# print(delta_ex)
# print(beam)
# print(err) # <1%

#Plot solution: Deformation and Von Mises

#Deformation plot
scale_factor=-1.0/beam #Scaling with 1/beam, the maximum deflection will be 1.0 in the plot. Flipped for conventional downward-bending visualization
# Extract u_x (even indices) and u_y (odd indices)
ux=u_approx[0::2]
uy=u_approx[1::2]

# Create the deformed nodes by applying the scale factor and adding the displacements to the original nodes
nodes_def=np.copy(nodes)
nodes_def[:, 0]+=ux*scale_factor
nodes_def[:, 1]+=uy*scale_factor

plt.figure(figsize=(10, 3))

# Draw the undeformed mesh
plt.triplot(nodes[:,0], nodes[:,1], elements, color='lightgray', linewidth=0.5, label='Undeformed')

# Draw the deformed mesh
plt.triplot(nodes_def[:,0], nodes_def[:,1], elements, color='blue', linewidth=0.8, label="Deformed (normalized)")

plt.title("Cantilever Beam: Deformed Configuration")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal') # Keeps the aspect ratio equal for x and y axes
plt.legend()
plt.show()

# Von Mises plot
n_elem=len(elements)
sigma_vm=np.zeros(n_elem)

for elem in range(n_elem):
    idxs=elements[elem,:]
    pt_triangle=nodes[idxs,:]
    
    # Collect the displacements for the current element
    u_elem=np.zeros(6)
    for i in range(3):
        u_elem[2*i]=u_approx[2*idxs[i]]
        u_elem[(2*i)+1] = u_approx[(2*idxs[i])+1]
        
    area=area_triangles(pt_triangle)
    B_elem=strain_displacement_matrix(pt_triangle, area)
    
    # sigma=[sigma_xx, sigma_yy, sigma_xy]
    eps=B_elem@u_elem
    sig=C@eps
    
    s_xx=sig[0]
    s_yy=sig[1]
    s_xy=sig[2]
    s_zz=nu*(s_xx+s_yy) # plane strain
    
    # Von Mises stress calculation for plane strain
    sigma_vm[elem]=np.sqrt(0.5*((s_xx-s_yy)**2+(s_yy-s_zz)**2+(s_zz-s_xx)**2)+3*s_xy**2)

plt.figure(figsize=(10, 3))
tpc = plt.tripcolor(nodes[:,0], nodes[:,1], elements, facecolors=sigma_vm, cmap='turbo', edgecolors='none')
plt.colorbar(tpc, label="Von Mises Stress $\sigma_{vM}$")
plt.title("Von Mises Stress Field (Undeformed Configuration)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.tight_layout()
plt.show()

#convergence analysis on reference solution (Timoshenko's beam theory)

#breakpoint()
nyrange=np.array([20,40,60,80])
err=np.zeros(len(nyrange))
for i,ny in enumerate(nyrange):
    nx=8*ny
    print('Running simulation for ny =',ny)
    beams, _, _, _=solve_beam(nx,ny)
    err[i]=abs(delta_ex-beams)

order=np.polyfit(np.log(nyrange), np.log(err), 1)[0] #degree 1, [0] extracts the order
print(f"Observed convergence order (vs ny): {-order:.2f}")

plt.loglog(nyrange,err,'*',label='numerical error')
plt.loglog(nyrange,err[-1]*(nyrange/nyrange[-1])**(-2),label='slope -2 (reference)')
plt.xlabel('ny (elements through beam height)')
plt.ylabel('|delta_ex - beam|')
plt.title(f"Error behaviour (observed order {-order:.2f})")
plt.legend()
plt.show()

#self-convergence analysis

# compute reference solution with a very fine mesh
beam_ref, _, _, _=solve_beam(8*160,160)
err=np.zeros(len(nyrange))
for i,ny in enumerate(nyrange):
    nx=8*ny
    print('Running simulation for ny =',ny)
    beams, _, _, _=solve_beam(nx,ny)
    err[i]=abs(beam_ref-beams)

order=np.polyfit(np.log(nyrange), np.log(err), 1)[0] #degree 1, [0] extracts the order
print(f"Observed convergence order (vs ny): {-order:.2f}")

plt.loglog(nyrange,err,'*',label='numerical error')
plt.loglog(nyrange,err[-1]*(nyrange/nyrange[-1])**(-2),label='slope -2 (reference)')
plt.xlabel('ny (elements through beam height)')
plt.ylabel('|beam_ref - beam|')
plt.title(f"Error behaviour (observed order {-order:.2f})")
plt.legend()
plt.show()