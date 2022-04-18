""" 
ArchimedesKitten Hood-Taylor on an unstructured mesh
Umberto Zerbinati & Fabio Credali

We vary the Cr(Credali number) and mesh size
"""

from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.internal import SnapShot
from ngsolve.utils import *
import numpy as np 
from time import sleep
import petsc4py.PETSc as psc
from netgen.geom2d import SplineGeometry
import ngsolve.ngs2petsc as n2p
from mpi4py import MPI

COMM = MPI.COMM_WORLD 

def Print(msg):
    if COMM.rank==0:
        print(msg)
sleep(3)


#Parameters
Crs = [1e-2,1,100]
N = 5
#H are the rows while Cr are the colums
ErrorPressure = np.zeros((N,len(Crs)));
ErrorVelocity = np.zeros((N,len(Crs)));
ErrorDiv = np.zeros((N,len(Crs)));


for CrIndex in range(len(Crs)):
    for hmaxIndex in range(N):

        Cr = Crs[CrIndex]
        hmax = 1/(2**(hmaxIndex+1))
        print("----| Mesh: {}, Cr {} |----".format(hmax,Cr))
        
        if COMM.rank == 0:
            geo = SplineGeometry()
            geo.AddRectangle((-2,-2),(2,2),bc="rect")
            ngmesh = geo.GenerateMesh(maxh=hmax).Distribute(COMM)
        else:
            ngmesh = netgen.meshing.Mesh.Receive(COMM)

        mesh = Mesh(ngmesh)

        Print("Mesh generated");

        W = VectorH1(mesh, order=4)
        V = VectorH1(mesh, order=1, dirichlet="rect")
        Q = H1(mesh, order=1)
        R = NumberSpace(mesh)

        u,v = V.TnT()
        p,q = Q.TnT()
        lam,mu = R.TnT()

        mp = BilinearForm(Q)
        mp += p*q*dx
        mp.Assemble()
        invmp = mp.mat.Inverse(inverse="umfpack")
        mesh.ngmesh.Refine()
        V.Update()
        Q.Update()

        prol = Q.Prolongation().Operator(1)
        iso = prol @ invmp @ prol.T

        a = BilinearForm(V)
        a += InnerProduct(grad(u),grad(v))*dx
        a.Assemble() 

        b = BilinearForm(trialspace=V,testspace=Q)
        b += (-1)*div(u)*q*dx;
        b.Assemble()

        l = BilinearForm(trialspace=Q,testspace=R)
        l += p*mu*dx;
        l.Assemble()


        f = LinearForm(V);
        f += CoefficientFunction((Cr + 24*((4-x**2)**2)*y-32*(x**2)*y*(4-y**2)+16*(4-x**2)*y*(4-y**2),32*x*(4-x**2)*(y**2)-16*x*(4-x**2)*(4-y**2)-24*x*(4-y**2)**2))*v*dx;
        f.Assemble()
        g = LinearForm(Q);
        g += 0*q*dx
        g.Assemble()
        h = LinearForm(R);
        h += 0*mu*dx
        h.Assemble()
        
        Print("Assembled the system");

        gfv = GridFunction(W)         
        gfu = GridFunction(V)
        gfp = GridFunction(Q)
        gfh = GridFunction(R)

        #Solving the problem
        Print("Solving the System");
        #STIFNESS
        K = BlockMatrix([[a.mat, b.mat.T,None],[b.mat, None,l.mat.T],[None,l.mat,None]])
        #PRECONDITIONER
        preI = Projector(mask=R.FreeDofs(), range=True)
        C = BlockMatrix([[a.mat.Inverse(V.FreeDofs()),None,None],[None,iso,None],[None,None,preI]])
        #RHS
        rhs = BlockVector ([f.vec, g.vec,h.vec])
        sol = BlockVector([gfu.vec, gfp.vec, gfh.vec])
        solvers.CG(mat=K, pre=C, rhs=rhs, sol=sol, printrates='\r', initialize=False, maxsteps=500);
        #Plotting the Solutions
        # VTOutput object

        # Exporting the results:
        velocity = CoefficientFunction((4*((4 - x**2)**2)*y*(4 - y**2),-4*x*(4 - x**2)*(4 - y**2)**2))
        gfv.Set(velocity)
        errV = Norm(gfv-gfu)**2+Norm(grad(gfv)-grad(gfu))**2
        ErrorVelocity[hmaxIndex,CrIndex] = Integrate(errV,mesh,order=3)**(1/2);

        pressure = CoefficientFunction(Cr*x);
        
        vtk = VTKOutput(ma=mesh,
                        coefs=[gfu,gfp,pressure],
                        names = ["velocity","pressure","exact"],
                        filename="./BPFigures/BP_US_{}_{}".format(str(Cr),str(hmax)),
                        subdivision=3)
        vtk.Do()
        sleep(3)
        
        errP = Norm(gfp-pressure)**2;
        ErrorPressure[hmaxIndex,CrIndex] = Integrate(errP,mesh,order=3)**(1/2);
        errDiv = Norm(div(gfu))**2
        ErrorDiv[hmaxIndex,CrIndex] = Integrate(errDiv,mesh,order=3)**(1/2);
        Print("Error in pressure {}, Error in velocity {}, Error in divergence".format(ErrorPressure[hmaxIndex,CrIndex],ErrorVelocity[hmaxIndex,CrIndex],ErrorDiv[hmaxIndex,CrIndex]))
np.save("BP_US_ErrorPressure",ErrorPressure) #Error is measured in L2
np.save("BP_US_ErrorVelocity",ErrorVelocity) #Error is measured in H1
np.save("BP_US_ErrorDiv",ErrorDiv) #Error is measured in H1
