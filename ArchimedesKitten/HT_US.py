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
        
        if COMM.rank == 0:
            geo = SplineGeometry()
            geo.AddRectangle((0,0),(1,1),bc="rect")
            ngmesh = geo.GenerateMesh(maxh=hmax).Distribute(COMM)
        else:
            ngmesh = netgen.meshing.Mesh.Receive(COMM)

        mesh = Mesh(ngmesh)

        Print("Mesh generated");

        V = VectorH1(mesh, order=2, dirichlet="rect")
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
        f += InnerProduct(CoefficientFunction((0,Ra*(1-y+3*(y**2)))),v)*dx;
        f += 0*q*dx;
        f.Assemble()
        
        Print("Assembled the system");

        gfu = GridFunction(X)
        
        #Solving the problem
        """
        psc_mat = n2p.CreatePETScMatrix(a.mat, X.FreeDofs())
        vecmap = n2p.VectorMapping (a.mat.row_pardofs, X.FreeDofs())
        psc_f, psc_u = psc_mat.createVecs()

        ksp = psc.KSP()
        ksp.create()
        ksp.setOperators(psc_mat)
        ksp.setType(psc.KSP.Type.CG)
        ksp.setNormType(psc.KSP.NormType.NORM_NATURAL)
        ksp.getPC().setType("gamg")
        ksp.setTolerances(rtol=1e-6, atol=0, divtol=1e16, max_it=400)

        vecmap.N2P(f.vec, psc_f)
        ksp.solve(psc_f, psc_u)
        vecmap.P2N(psc_u, gfu.vec)
        """
        Print("Solving the System");
        gfu.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
        #Plotting the Solutions
        # VTKOutput object

        # Exporting the results:

        errV = Norm(gfu.components[0])**2+Norm(grad(gfu.components[0]))**2
        ErrorVelocity[hmaxIndex,RaIndex] = Integrate(errV**(1/2),mesh,order=3);

        pressure = CoefficientFunction(Ra*(y**3-0.5*y**2+y-7/12))
        
        vtk = VTKOutput(ma=mesh,
                        coefs=[gfu.components[0],gfu.components[1],pressure],
                        names = ["velocity","pressure","exact"],
                        filename="./HTFigures/HT_US_{}_{}".format(str(Ra),str(hmax)),
                        subdivision=3)
        vtk.Do()
        sleep(3)
        
        errP = Norm(gfu.components[1]-pressure);
        ErrorPressure[hmaxIndex,RaIndex] = Integrate(errP**(1/2),mesh,order=3);
        Print("Error in pressure {}, Error in velocity {}".format(ErrorPressure[hmaxIndex,RaIndex],ErrorVelocity[hmaxIndex,RaIndex]))
np.save("HT_US_ErrorPressure",ErrorPressure) #Error is measured in L2
np.save("HT_US_ErrorVelocity",ErrorVelocity) #Error is measured in H1 
