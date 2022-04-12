"""
ArchimedesKitten Hood-Taylor on an unstructured mesh
Umberto Zerbinati & Fabio Credali

We vary the Ra (Rayleigh number) and mesh size
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
Ras = [1,100,1e4]
N = 6

#H are the rows while Ra are the colums
ErrorPressure = np.zeros((N,len(Ras)));
ErrorVelocity = np.zeros((N,len(Ras)));


for RaIndex in range(len(Ras)):
    for hmaxIndex in range(N):

        Ra = Ras[RaIndex]
        hmax = 1/(2**(hmaxIndex+1))
        print("----| Mesh: {}, Ra {} |----".format(hmax,Ra))
        
        geo = SplineGeometry()
        geo.AddRectangle((0,0),(1,1),bc="rect")
        ngmesh = geo.GenerateMesh(maxh=hmax)
        mesh = Mesh(ngmesh)

        Print("Mesh generated");
        """
        Usual Stokes
        ------------
        grad(u)*grad(v) + div(v)*p = fv
        div(u)*q = 0
        null average condition
        """

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

        #STIFNESS
        K = BlockMatrix([[a.mat, b.mat.T,None],[b.mat, None,l.mat.T],[None,l.mat,None]])
        #PRECONDITIONER
        preI = Projector(mask=R.FreeDofs(), range=True)
        C = BlockMatrix([[a.mat.Inverse(V.FreeDofs()),None,None],[None,iso,None],[None,None,preI]])
        #RHS
        f = LinearForm(V);
        f += InnerProduct(CoefficientFunction((0,Ra*(1-y+3*(y**2)))),v)*dx;
        f.Assemble()
        g = LinearForm(Q);
        g += 0*q*dx
        g.Assemble()
        h = LinearForm(R);
        h += 0*mu*dx
        h.Assemble()
        
        Print("Assembled the system");

        gfu = GridFunction(V)
        gfp = GridFunction(Q)
        gfh = GridFunction(R)
        
        #Solving the problem
        Print("Solving the System");
        rhs = BlockVector ([f.vec, g.vec,h.vec])
        sol = BlockVector([gfu.vec, gfp.vec, gfh.vec])
        solvers.CG(mat=K, pre=C, rhs=rhs, sol=sol, printrates='\r', initialize=False, maxsteps=500);

        #Plotting the Solutions
        # VTKOutput object

        # Exporting the results:

        errV = Norm(gfu)**2+Norm(grad(gfu))**2
        ErrorVelocity[hmaxIndex,RaIndex] = Integrate(errV,mesh,order=3)**(1/2);

        pressure = CoefficientFunction(Ra*(y**3-0.5*y**2+y-7/12))
        
        vtk = VTKOutput(ma=mesh,
                        coefs=[gfu,gfp,pressure],
                        names = ["velocity","pressure","exact"],
                        filename="./BPFigures/BP_US_{}_{}".format(str(Ra),str(hmax)),
                        subdivision=3)
        vtk.Do()
        sleep(3)
        
        errP = Norm(gfp-pressure)**2;
        ErrorPressure[hmaxIndex,RaIndex] = Integrate(errP,mesh,order=3)**(1/2);
        Print("Error in pressure {}, Error in velocity {}".format(ErrorPressure[hmaxIndex,RaIndex],ErrorVelocity[hmaxIndex,RaIndex]))
np.save("BP_US_ErrorPressure",ErrorPressure) #Error is measured in L2
np.save("BP_US_ErrorVelocity",ErrorVelocity) #Error is measured in H1 
