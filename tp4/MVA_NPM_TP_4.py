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
        i = -(2*i - nlig)/nlig
        j = (2*j - ncol)/nlig

        wo = np.stack((-i, -j, 1.5 * np.ones_like(i)), axis=-1)
        wo = wo / np.linalg.norm(wo, axis=-1, keepdims=True)

        for light in lights:
            wi = light.position - np.stack((j, i, np.zeros_like(i)), axis=-1)
            wi = wi / np.linalg.norm(wi, axis=-1, keepdims=True)

            f = self.f(wo, wi, normal)

            Li = light.intensity * light.color
            nwi = np.sum(normal*wi,axis=-1,keepdims=True)
            res = f * Li * nwi
            # low = np.percentile(res, 0)
            # high = np.percentile(res, 100)
            render += np.clip(f*Li*nwi, 0, None) #max(0,low), high)
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
    def __init__(self, alpha, specular, metal):
        self.alpha = alpha
        self.roughness = np.sqrt(self.alpha)
        self.k = (self.roughness + 1)**2/8
        self.F0 = metal
        self.spec = np.array(specular)
    

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
        F = self.spec * (self.F0 + (1 - self.F0)*2**((c1*oh + c2)*oh))

        F[(nwi <= 0).squeeze()] = 0

        return D*F*G/(4 * nwi * nwo)
        

    
class LightSource():
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity
    
    def set_position(self, position):
        self.position = np.array(position)

if __name__ == "__main__":
    imagefile = "normal.png"
    renderfile = "render.png"

    # Read normal image
    normalimage = plt.imread(imagefile)[:,:,:3]
    mask = normalimage.max(axis=2) <= 4/255
    normalimage = 2 * normalimage -1
    normalimage = normalimage / np.linalg.norm(normalimage, axis=-1, keepdims=True)
    normalimage[mask] = 0
    
    # Display normal image
    if False:
        plt.imshow(normalimage)
        plt.show()

    # Define materials and light source
    # lambert = Lambert([0.3, 0.8, 0.5], 3)
    lambert = Lambert([0.80,0.42,0.42],1)

    blinn = BlinnPhong([0.80,0.42,0.42],5)

    cook = Cook(0.5, [0.8,0.42,0.42], 1)
    material = cook

    light = LightSource([0.,0,0.5], [1.,1.,1.], 50)


    fig, ax = plt.subplots()
    render = material.shade(normalimage[:,:,:3], [light])
    #render = render / render.max()

    im = plt.imshow(render)
    nlig,ncol,_ = render.shape

    from matplotlib.widgets import Slider
    #axx = plt.axes([0.25, 0.1, 0.65, 0.03])
    #axy = plt.axes([0.25, 0.15, 0.65, 0.03])
    #lx = Slider(axx,"x",-1.,1.,0)
    #ly = Slider(axy,"y",-1.,1.,0)

    def onmotion(event):
        if event.xdata is None or event.ydata is None: return
        x = (2*event.xdata-ncol)/nlig
        y = -(2*event.ydata-nlig)/nlig
        light.set_position([x,y,0.5])
        render = material.shade(normalimage[:,:,:3],[light])
        #render = render / render.max()
        im.set_data(render)
        plt.draw()

    cid = fig.canvas.mpl_connect('motion_notify_event', onmotion)

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

    