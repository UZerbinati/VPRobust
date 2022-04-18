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

        V = VectorH1(mesh, order=2, dirichlet="rect")
        W = VectorH1(mesh, order=4)
        Q = H1(mesh, order=1)
        R = NumberSpace(mesh)
        X = V*Q*R

        u,p,lam = X.TrialFunction()
        v,q,mu = X.TestFunction()

        a = BilinearForm(X)

        """
        Usual Stokes
        ------------
        grad(u)*grad(v) + div(v)*p = fv
        div(u)*q = 0
        null average condition
        """

        a += (InnerProduct(grad(u),grad(v))+div(u)*q-div(v)*p)*dx
        a += (lam*q+mu*p) * dx
        a.Assemble() 

        f = LinearForm(X);
        f += CoefficientFunction((Cr + 24*((4-x**2)**2)*y-32*(x**2)*y*(4-y**2)+16*(4-x**2)*y*(4-y**2),32*x*(4-x**2)*(y**2)-16*x*(4-x**2)*(4-y**2)-24*x*(4-y**2)**2))*v*dx;
        f += 0*q*dx;
        f.Assemble()
        
        Print("Assembled the system");

        gfu = GridFunction(X)         
        gfv = GridFunction(W)         
        
        #Solving the problem
        Print("Solving the System");
        gfu.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
        #Plotting the Solutions
        # VTOutput object

        # Exporting the results:
        velocity = CoefficientFunction((4*((4 - x**2)**2)*y*(4 - y**2),-4*x*(4 - x**2)*(4 - y**2)**2))
        gfv.Set(velocity)
        errV = Norm(gfv-gfu.components[0])**2+Norm(grad(gfv)-grad(gfu.components[0]))**2
        ErrorVelocity[hmaxIndex,CrIndex] = Integrate(errV,mesh,order=3)**(1/2);

        pressure = CoefficientFunction(Cr*x);
        
        vtk = VTKOutput(ma=mesh,
                        coefs=[gfu.components[0],gfu.components[1],pressure],
                        names = ["velocity","pressure","exact"],
                        filename="./HTFigures/HT_US_{}_{}".format(str(Cr),str(hmax)),
                        subdivision=3)
        vtk.Do()
        sleep(3)
        
        errP = Norm(gfu.components[1]-pressure)**2;
        ErrorPressure[hmaxIndex,CrIndex] = Integrate(errP,mesh,order=3)**(1/2);
        Print("Error in pressure {}, Error in velocity {}".format(ErrorPressure[hmaxIndex,CrIndex],ErrorVelocity[hmaxIndex,CrIndex]))
np.save("HT_US_ErrorPressure",ErrorPressure) #Error is measured in L2
np.save("HT_US_ErrorVelocity",ErrorVelocity) #Error is measured in H1 
