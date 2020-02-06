import numpy as np
import matplotlib.pyplot as plt

class Material():
    def __init__(self):
        pass

    def f(self, wi, wo, n):
        pass

    def __add__(self, other):
        result = Material()
        def f(wi, wo, n):
            return self.f(wi,wo,n) + other.f(wi,wo,n)
        result.f = f 
        return result

class Lambert(Material):
    def __init__(self, albedo, diffuse):
        super().__init__()
        self.albedo = np.array(albedo)
        self.diffuse = diffuse
    
    def f(self, wo, wi, n):
        return self.albedo * self.diffuse / np.pi

class BlinnPhong(Material):
    def __init__(self,diffuse,shine):
        super().__init__()
        self.diffuse = np.array(diffuse)
        self.shine = shine

    def f(self, wo, wi, n):
        mid = wo + wi
        wh = mid / np.linalg.norm(mid)
        return self.diffuse * np.dot(n,wh)**self.shine

class Cook(Material):
    def __init__(self, roughness, metal, specular):
        self.roughness = roughness
        self.alpha_sq = self.roughness ** 4
        self.metal = metal
        self.spec = np.array(specular)
    
    def f(self, wo, wi, n):
        mid = wo + wi
        wh = mid / np.linalg.norm(mid)

        D = self.alpha_sq / (np.pi *(np.dot(n,wh)**2*(self.alpha_sq-1)+1)**2)
        
        k = (self.roughness + 1)**2 / 8
        nwi = np.dot(n,wi)
        Gi = nwi/(nwi*(1-k)+k)

        nwo = np.dot(n,wo)
        Go = nwo/(nwo*(1-k)+k)

        G = Gi * Go

        c1 = -5.55473
        c2 = -6.98316
        ih = np.dot(wh, wi)
        F = self.metal + (1 - self.metal)*2**((c1*ih + c2)*ih)

        return D*F*G/(4 * np.dot(n,wi) * np.dot(n,wo))
        
        

    
class LightSource():
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

def shade(normalimage, material, lightsources):
    nlig, ncol, _ = normalimage.shape
    render = np.zeros((nlig, ncol, 3))

    for i, j in np.ndindex(nlig, ncol):
        if False and np.all(normalimage[i,j,:3] <= 3/255):
            render[i,j] = [0.,0.,0]
        else:
            for ls in lightsources:
                n = 2 * (normalimage[i,j,:3]-0.5)
                n = n/np.linalg.norm(n)
                x = (2*j - ncol)/ncol
                y = -(2*i - nlig)/ncol
                wo = np.array([-x, -y ,1.5])
                wo = wo/np.linalg.norm(wo)
                wi = ls.position - np.array([x,y,0])
                wi = wi/np.linalg.norm(wi)
                f = material.f(wo, wi, n)
                Li =  ls.intensity * ls.color
                render[i,j] += np.clip(f * Li * np.dot(n, wi), 0, None)
    return render 

if __name__ == "__main__":
    imagefile = "normal.png"
    renderfile = "render.png"

    # Read normal image
    normalimage = plt.imread(imagefile)

    # Display normal image
    if True:
        plt.imshow(normalimage)
        plt.show()

    # Define materials and light source
    # lambert = Lambert([0.3, 0.8, 0.5], 3)
    lambert = Lambert([1.,1.,1.],2)
    blinn = BlinnPhong([1,1,1.],2)
    cook = Cook(0.7, 0.5, [0,0,1])
    material = lambert + blinn + cook

    light_source1 = LightSource([0,1.,1.], [1.,1.,1.], 0.75)
    light_source2 = LightSource([1.,0.,1.], [1.,0.,0.], 0.4)
    light_source3 = LightSource([0,-1.,1.], [0,0,1.], 0.4)

    render = shade(normalimage, material, [light_source1, light_source2, light_source3])
    
    plt.imshow(render)
    plt.show()

    print(render.min(), render.max())
    plt.imsave(renderfile, render)