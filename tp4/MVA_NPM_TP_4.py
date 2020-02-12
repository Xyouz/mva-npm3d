import numpy as np
import matplotlib.pyplot as plt

class Material():
    def __init__(self):
        pass

    def f(self, wi, wo, n):
        pass

    def shade(self, normal, lights):
        nlig, ncol, _ = normalimage.shape
        render = np.zeros((nlig, ncol, 3))

        i,j = np.meshgrid(range(nlig), range(ncol),indexing='ij')
        i = -(2*i - nlig)/ncol
        j = (2*j - ncol)/ncol

        wo = np.stack((-j, -i, 1.5 * np.ones_like(i)), axis=-1)
        
        for light in lights:
            wi = light.position - np.stack((-j, -i, np.zeros_like(i)), axis=-1)

            f = self.f(wo, wi, normal)

            Li = light.intensity * light.color
            nwi = np.sum(normal*wi,axis=-1,keepdims=True)
            render += np.clip(f*Li*nwi, 0, None)
        return render 

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
        wh = mid / np.linalg.norm(mid, axis=-1, keepdims=True)
        nwh = np.sum(n*wh, axis=-1, keepdims=True)
        return self.diffuse * nwh**self.shine

class Cook(Material):
    def __init__(self, alpha, specular):
        self.alpha = alpha
        self.roughness = np.sqrt(self.alpha)
        self.k = (self.roughness + 1)**2/8
        self.F0 = np.array(specular)
    

    def f(self, wo, wi, n):
        mid = wo + wi
        wh = mid / np.linalg.norm(mid)

        nwh = np.sum(n*wh, axis=-1, keepdims=True)
        nwi = np.sum(n*wi, axis=-1, keepdims=True)
        nwo = np.sum(n*wo, axis=-1, keepdims=True)

        D = self.alpha**2 / (np.pi *(nwh**2*(self.alpha**2-1)+1)**2)
        
        k = (self.roughness + 1)**2 / 8
        
        Gi = nwi/(nwi*(1-k)+k)
        Go = nwo/(nwo*(1-k)+k)

        G = Gi * Go

        c1 = -5.55473
        c2 = -6.98316
        oh = np.sum(wh * wo, axis=-1, keepdims=True)
        F = self.F0 + (1 - self.F0)*2**((c1*oh + c2)*oh)

        return D*F*G/(4 * nwi * nwo)
        

    
class LightSource():
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

def shade(normalimage, material, lightsources):
    nlig, ncol, _ = normalimage.shape
    render = np.zeros((nlig, ncol, 3))

    for i, j in np.ndindex(nlig, ncol):
        for ls in lightsources:
            n =  (normalimage[i,j,:3]-0.5)
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
    if False:
        plt.imshow(normalimage)
        plt.show()

    # Define materials and light source
    # lambert = Lambert([0.3, 0.8, 0.5], 3)
    lambert = Lambert([0.70,0.62,0.62],1)
    lambert2 = Lambert([0.70,0.62,0.62],0.6)

    blinn = BlinnPhong([0.70,0.62,0.62],0.75)

    cook = Cook(0.3, [0.70,0.62,0.62])
    material = blinn

    light_source1 = LightSource([0.,0,1.0], [1.,1.,1.], 0.5)
    light_source2 = LightSource([1.,0.,1.], [1.,0.1,0.3], 5)
    light_source3 = LightSource([-1,0.,1.], [0.1,0.4,1.], 5)

    fig, ax = plt.subplots()
    render = lambert.shade(normalimage[:,:,:3], [light_source1])
    im = plt.imshow(render)
    nlig,ncol,_ = render.shape

    from matplotlib.widgets import Slider
    axx = plt.axes([0.25, 0.1, 0.65, 0.03])
    axy = plt.axes([0.25, 0.15, 0.65, 0.03])
    lx = Slider(axx,"x",-1.,1.,0)
    ly = Slider(axy,"y",-1.,1.,0)

    def onclick(event):
        x = 5*(event.x-ncol/2)/ncol
        y = -5*(event.y-nlig/2)/ncol
        print(x,y)
        light = LightSource([x,y,1.], [1.,1.,1.], 1)
        render = lambert.shade(normalimage[:,:,:3],[light])
        im.set_data(render)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    #lx.on_changed(update)
    plt.show()
    # from  tqdm import tqdm
    # files = []
    # for i, x in tqdm(enumerate(np.linspace(-1,1,100))):
    #     light = LightSource([x,x,1.0], [1.,1.,1.], 0.75)
    #     render = blinn.shade(normalimage[:,:,:3], [light])#, light_source2, light_source3])
    #     plt.cla()
    #     plt.imshow(render)
    #     fname = "tmp-{}.png".format(i)
    #     plt.savefig(fname)
    #     files.append(fname)
    
    # import subprocess
    # subprocess.call("mencoder 'mf://tmp-*.png' -mf type=png:fps=10 -ovc lavc "
    #             "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)

    # for fname in files:
    #     os.remove(fname)

    # render = shade(normalimage[:,:,:3],cook ,[light_source1])
    # plt.imshow(render)
    # plt.show()

    