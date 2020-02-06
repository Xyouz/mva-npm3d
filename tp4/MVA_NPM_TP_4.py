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
        pass
    
    def f(self, wo, wi, n):
        pass
    
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
                x = (2*j - ncol)/ncol
                y = -(2*i - nlig)/ncol
                wo = np.array([-x, -y ,1.5])
                wi = ls.position - np.array([x,y,0])
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
    if False:
        plt.imshow(normalimage)
        plt.show()

    # Define materials and light source
    # lambert = Lambert([0.3, 0.8, 0.5], 3)
    lambert = Lambert([1.,1.,1.],2)
    blinn = BlinnPhong([1,1,1.],2)
    material = lambert + blinn

    light_source1 = LightSource([0,1.,1.0], [1.,1.,1.], 0.5)
    # light_source2 = LightSource([1.,0.,1.], [1.,0.,0.], 0.75)
    # light_source3 = LightSource([0,0.,0], [1.,1,1.], 1)

    render = shade(normalimage, material, [light_source1])#, light_source2])#, light_source3])

    # print(render.min(), render.max())
    plt.imshow(render)
    plt.show()

    plt.imsave(renderfile, render)