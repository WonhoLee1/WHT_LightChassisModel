import jax
import jax.numpy as jnp

@jax.jit
def _get_shape_functions(xi, eta):
    return 0.25 * jnp.array([
        (1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)
    ])

@jax.jit
def _get_shape_derivatives(xi, eta):
    dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dN_deta = 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    return dN_dxi, dN_deta

@jax.jit
def _get_mb_matrices(xi, eta, coords):
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:,0]), jnp.dot(dN_dxi, coords[:,1])],
        [jnp.dot(dN_deta, coords[:,0]), jnp.dot(dN_deta, coords[:,1])]
    ])
    detJ = jnp.linalg.det(J)
    invJ = jnp.linalg.inv(J)
    
    dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
    dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
    
    Bm = jnp.zeros((3, 24))
    for i in range(4):
        Bm = Bm.at[0, 6*i].set(dN_dx[i])
        Bm = Bm.at[1, 6*i+1].set(dN_dy[i])
        Bm = Bm.at[2, 6*i].set(dN_dy[i])
        Bm = Bm.at[2, 6*i+1].set(dN_dx[i])
        
    Bb = jnp.zeros((3, 24))
    for i in range(4):
        Bb = Bb.at[0, 6*i+4].set(dN_dx[i])
        Bb = Bb.at[1, 6*i+3].set(-dN_dy[i])
        Bb = Bb.at[2, 6*i+4].set(dN_dy[i])
        Bb = Bb.at[2, 6*i+3].set(-dN_dx[i])
        
    return Bm, Bb, detJ

@jax.jit
def _get_Bs_raw_13(xi, eta, coords):
    Bs_row = jnp.zeros(24)
    N = _get_shape_functions(xi, eta)
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:, 0]), jnp.dot(dN_dxi, coords[:, 1])],
        [jnp.dot(dN_deta, coords[:, 0]), jnp.dot(dN_deta, coords[:, 1])]
    ])
    invJ = jnp.linalg.inv(J)
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    for i in range(4):
        Bs_row = Bs_row.at[6*i+2].set(dN_dx[i])
        Bs_row = Bs_row.at[6*i+4].set(N[i])
    return Bs_row

@jax.jit
def _get_Bs_raw_23(xi, eta, coords):
    Bs_row = jnp.zeros(24)
    N = _get_shape_functions(xi, eta)
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:, 0]), jnp.dot(dN_dxi, coords[:, 1])],
        [jnp.dot(dN_deta, coords[:, 0]), jnp.dot(dN_deta, coords[:, 1])]
    ])
    invJ = jnp.linalg.inv(J)
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    for i in range(4):
        Bs_row = Bs_row.at[6*i+2].set(dN_dy[i])
        Bs_row = Bs_row.at[6*i+3].set(-N[i])
    return Bs_row

@jax.jit
def _element_K_mitc4_plus_jax(c1, c2, c3, c4, t, E, nu):
    v12 = c2 - c1
    v14 = c4 - c1
    x_loc = v12 / (jnp.linalg.norm(v12) + 1e-12)
    z_raw = jnp.cross(x_loc, v14)
    z_loc = z_raw / (jnp.linalg.norm(z_raw) + 1e-12)
    y_loc = jnp.cross(z_loc, x_loc)
    T_mat = jnp.stack([x_loc, y_loc, z_loc], axis=0)
    
    p2 = c2 - c1; p3 = c3 - c1; p4 = c4 - c1
    coords_2d = jnp.array([
        [0.0, 0.0],
        [jnp.dot(p2, x_loc), jnp.dot(p2, y_loc)],
        [jnp.dot(p3, x_loc), jnp.dot(p3, y_loc)],
        [jnp.dot(p4, x_loc), jnp.dot(p4, y_loc)]
    ])
    
    G = E / (2.0 * (1.0 + nu))
    k_shear = 5.0 / 6.0
    inv_nu2 = 1.0 / (1.0 - nu**2)
    Dm = (E * t * inv_nu2) * jnp.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Db = (E * t**3 / 12.0 * inv_nu2) * jnp.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Ds = (k_shear * G * t) * jnp.eye(2)
    
    K_loc = jnp.zeros((24, 24))
    K_drill = jnp.zeros((24, 24))
    gp = jnp.array([-1.0/jnp.sqrt(3), 1.0/jnp.sqrt(3)])
    Ktt = 1.0 * G * t 

    for xi_g in gp:
        for eta_g in gp:
            Bm, Bb, detJ = _get_mb_matrices(xi_g, eta_g, coords_2d)
            
            N_list = _get_shape_functions(xi_g, eta_g)
            dN_dxi, dN_deta = _get_shape_derivatives(xi_g, eta_g)
            J = jnp.array([
                [jnp.dot(dN_dxi, coords_2d[:,0]), jnp.dot(dN_dxi, coords_2d[:,1])],
                [jnp.dot(dN_deta, coords_2d[:,0]), jnp.dot(dN_deta, coords_2d[:,1])]
            ])
            invJ = jnp.linalg.inv(J)
            detJ2 = jnp.linalg.det(J)
            
            valid_mask = jnp.where(detJ2 < 1e-11, 0.0, 1.0)
            
            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
            
            Bd = jnp.zeros((1, 24))
            for i in range(4):
                Bd = Bd.at[0, 6*i].set(-0.5 * dN_dy[i])
                Bd = Bd.at[0, 6*i+1].set(0.5 * dN_dx[i])
                Bd = Bd.at[0, 6*i+5].set(-N_list[i])
            
            K_drill += valid_mask * ((Bd.T @ Bd) * Ktt * detJ2)

            Bs13_A = _get_Bs_raw_13(0.0, -1.0, coords_2d)
            Bs13_B = _get_Bs_raw_13(0.0,  1.0, coords_2d)
            Bs13 = 0.5 * (1.0 - eta_g) * Bs13_A + 0.5 * (1.0 + eta_g) * Bs13_B
            
            Bs23_C = _get_Bs_raw_23(-1.0, 0.0, coords_2d)
            Bs23_D = _get_Bs_raw_23( 1.0, 0.0, coords_2d)
            Bs23 = 0.5 * (1.0 - xi_g) * Bs23_C + 0.5 * (1.0 + xi_g) * Bs23_D
            
            Bs = jnp.vstack([Bs13, Bs23])
            
            K_loc += valid_mask * ((Bm.T @ Dm @ Bm + Bb.T @ Db @ Bb + Bs.T @ Ds @ Bs) * detJ)

    K_loc += K_drill
        
    T_24 = jnp.kron(jnp.eye(8), T_mat)
    return T_24.T @ K_loc @ T_24
