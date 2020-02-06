import numpy as np
import matplotlib.pyplot as plt

class Material():
    def __init__(self):
        pass

    def f(self, wi, wo, n):
        pass

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
        self.diffuse = diffuse
        self.shine = shine

    def f(self, wo, wi, n):
        mid = wo + wi
        wh = mid / np.linalg.norm(mid)
        return self.diffuse * np.dot(n,wh)**self.shine
    
class LightSource():
    def __init__(self, position, color, intensity):
        self.position = np.array(position).reshape(3,1)
        self.color = np.array(color)
        self.intensity = intensity

def shade(normalimage, material, lightsources):
    nlig, ncol, _ = normalimage.shape
    render = np.zeros((nlig, ncol, 3))

    for i, j in np.ndindex(nlig, ncol):
        if np.all(normalimage[i,j,:3] <= 3/255):
            render[i,j] = [0.,0.,0]
        else:
            for ls in lightsources:
                n = normalimage[i,j,:3] - 0.5
                wo = np.array([-(2*i - nlig)/nlig, -(2*j - ncol)/ncol,-1]).reshape(3,1)
                wi = ls.position + wo
                f = material.f(wo, wi, n)
                Li =  ls.intensity * ls.color
                render[i,j] += f * Li * np.dot(n, wi)
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
    material = Lambert([1, 1., 1], 3)
    light_source1 = LightSource([1,1.,1], [1.,0.,0.], 1)
    light_source2 = LightSource([1,-1.,1], [0.,0.,1.], 0.5)

    render = shade(normalimage, material, [light_source1, light_source2])

    plt.imshow(render)
    plt.show()

    plt.imsave(renderfile, render)