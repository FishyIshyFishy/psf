#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, affine_transform

def generate_aniso_gaussian(shape, voxel_size, sigmas, direction):
    # build orthonormal frame {e1,e2,e3}
    e1 = np.array(direction, float);  e1 /= np.linalg.norm(e1)
    tmp = np.array([1,0,0]) if abs(e1[0])<0.9 else np.array([0,1,0])
    e2 = np.cross(e1, tmp); e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    # grid in phys coords
    Z, Y, X = shape
    zz = (np.arange(Z)-(Z-1)/2)*voxel_size[0]
    yy = (np.arange(Y)-(Y-1)/2)*voxel_size[1]
    xx = (np.arange(X)-(X-1)/2)*voxel_size[2]
    Zg,Yg,Xg = np.meshgrid(zz,yy,xx, indexing='ij')
    pts = np.stack((Zg, Yg, Xg), axis=-1).reshape(-1,3)
    # project into e1,e2,e3
    M = np.vstack((e1,e2,e3))
    coords = pts @ M.T
    s1,s2,s3 = sigmas
    G = np.exp(-((coords[:,0]**2)/(2*s1**2)
                + (coords[:,1]**2)/(2*s2**2)
                + (coords[:,2]**2)/(2*s3**2)))
    return G.reshape(Z,Y,X)

def mip_perp(vol, vox_ds, direction, half=5, N=201, L=5, samples=201):
    d1 = np.array(direction,float); d1/=np.linalg.norm(d1)
    tmp = np.array([1,0,0]) if abs(d1[0])<0.9 else np.array([0,1,0])
    d2 = np.cross(d1,tmp); d2/=np.linalg.norm(d2)
    d3 = np.cross(d1,d2)
    U = np.linspace(-half,half,N); V=U; T=np.linspace(-L,L,samples)
    center = 0.5*(np.array(vol.shape)-1)*np.array(vox_ds)
    MIP = np.zeros((N,N),float)
    for i,u in enumerate(U):
        for j,v in enumerate(V):
            pts = (center[None]+u*d2[None]+v*d3[None]
                   + T[:,None]*d1[None])
            vox = pts/np.array(vox_ds)
            coords = np.stack([vox[:,k] for k in range(3)],0)
            MIP[j,i] = map_coordinates(vol, coords, order=1).max()
    return MIP, U, V

def rotate_and_mip(vol, direction, vox_ds):
    d1 = np.array(direction,float); d1/=np.linalg.norm(d1)
    tmp = np.array([1,0,0]) if abs(d1[0])<0.9 else np.array([0,1,0])
    d2 = np.cross(d1,tmp); d2/=np.linalg.norm(d2)
    d3 = np.cross(d1,d2)
    R = np.vstack((d2,d3,d1))
    center = 0.5*(np.array(vol.shape)-1)
    offset = center - R.dot(center)
    rot = affine_transform(vol, R, offset=offset, order=1, mode='nearest')
    return rot.max(axis=0)

def main():
    shape = (80,80,80)
    vox   = (0.1,0.1,0.1)
    # elongate along 45° in XY (σ_along=2, perpendicular σ=1)
    direction = [1,1,0]
    blob = generate_aniso_gaussian(shape, vox, (2.0,1.0,1.0), direction)

    mip2d, U, V = mip_perp(blob, vox, direction)
    mip_xy      = rotate_and_mip(blob, direction, vox)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
    im1=ax1.imshow(mip2d, origin='lower',
        extent=[U[0],U[-1],V[0],V[-1]], cmap='magma',aspect='equal')
    ax1.set_title('Custom MIP ⟂ 45°'); ax1.set_xlabel('d2 (µm)'); ax1.set_ylabel('d3 (µm)')
    fig.colorbar(im1,ax=ax1)

    ex = (-(blob.shape[2]-1)/2*vox[2], (blob.shape[2]-1)/2*vox[2])
    ey = (-(blob.shape[1]-1)/2*vox[1], (blob.shape[1]-1)/2*vox[1])
    im2=ax2.imshow(mip_xy,origin='lower',extent=[ex[0],ex[1],ey[0],ey[1]],
                   cmap='magma',aspect='equal')
    ax2.set_title('Rotated→XY MIP'); ax2.set_xlabel('X (µm)'); ax2.set_ylabel('Y (µm)')
    fig.colorbar(im2,ax=ax2)
    plt.tight_layout(); plt.show()

if __name__=='__main__':
    main()
