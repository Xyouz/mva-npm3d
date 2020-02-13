import numpy as np
import matplotlib.pyplot as plt

import argparse
from  tqdm import tqdm
import subprocess
import os
import random

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

        return D*F*G/(4 * nwi * nwo + 1e-10)
        

    
class LightSource():
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity
    
    def set_position(self, position):
        self.position = np.array(position)

def shade( normal, material, lights):
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

        f = material.f(wo, wi, normal)

        Li = light.intensity * light.color
        nwi = np.sum(normal*wi,axis=-1,keepdims=True)
        res = f * Li * nwi
        low = np.percentile(res, 0)
        high = np.percentile(res, 100)
        render += np.clip(f*Li*nwi, max(0,low), high)
    return render

def clip_render(render, perc=5):
    low = np.percentile(render, perc)
    high = np.percentile(render, 100-perc)

    render = np.clip(render, low, high)
    render -= low
    render /= (high - low)
    return render

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--normal", help="normal map file", type=str, default="normal.png")
    parser.add_argument("-o", "--output", help="output file", type=str, default="render.png")

    parser.add_argument("-m", "--mode", help="""Select what the script does:\n
      simple : take a normal map and renders before saving the result\n
      video : take a normal map and save a video output with multiple lighting conditions\n
      interactif : take a normal map and allow the interactive placement of light sources""",
      type=str, default="simple", choices=["simple", "video", "interactif"])

    parser.add_argument("-l", "--lambert", help="Use the Lambert model", action="store_true")
    parser.add_argument("-b", "--blinn", help="Use the Blinn-Phong model", action="store_true")
    parser.add_argument("-c", "--cook", help="Use the Cook-Torrance model", action="store_true", default=True)

    parser.add_argument("--n-lights", help="Number of lights for the simple mode", type=int, default=1)
    parser.add_argument("--n-frames", help="The number of frames to compute in video mode", type=int, default=100)
    parser.add_argument("--fps", help="The number of fps to output in video mode", type=int, default=20)

    args = parser.parse_args()

    imagefile = args.normal
    renderfile = args.output

    
    # Define materials
    if args.cook:
        LMin,  LMax = 45, 55
        material = Cook(5, [0.8,0.42,0.42], 0.02)
    elif args.blinn:
        LMin,  LMax = 0.5, 1
        material = BlinnPhong([0.80,0.42,0.42],5)
    elif args.lambert:
        LMin,  LMax = 0.5, 0.75
        material = Lambert([0.42,0.42,0.42],2)
    else:
        print("No material selected")
    
    # Display normal image
    if False:
        plt.imshow(normalimage)
        plt.show()

    light_intensity = lambda : LMin + (LMax - LMin) * np.random.random(1)
    light_color = lambda : 0.5 + 0.5 *np.random.random(3)
    light_pos = lambda : [-1 + 2* random.random(),-1 + 2* random.random(),0.5]

    if args.mode == "simple":
        # Read normal image
        normalimage = plt.imread(imagefile)[:,:,:3]
        plt.imshow(normalimage)
        plt.show()

        mask = normalimage.max(axis=2) <= 4/255
        normalimage = 2 * normalimage - 1
        normalimage = normalimage / np.linalg.norm(normalimage, axis=-1, keepdims=True)
        normalimage[mask] = 0

        lights = [LightSource(light_pos(), light_color(), light_intensity()/args.n_lights) for _ in range(args.n_lights)]

        render = shade(normalimage, material, lights)
        render = clip_render(render)

        plt.imshow(render)
        plt.imsave(renderfile, render)
        plt.show()
    
    elif args.mode == "interactif":
        # Read normal image
        normalimage = plt.imread(imagefile)[:,:,:3]
        mask = normalimage.max(axis=2) <= 4/255
        normalimage = 2 * normalimage - 1
        normalimage = normalimage / np.linalg.norm(normalimage, axis=-1, keepdims=True)
        normalimage[mask] = 0

        lights = [LightSource([0.,0.,0.5], light_color(), light_intensity())]

        fig, ax = plt.subplots()
        render = shade(normalimage, material, lights)
        render = clip_render(render)

        im = plt.imshow(render)
        nlig,ncol,_ = render.shape

        def onmotion(event):
            """
            Changes the last light position
            """
            if event.xdata is None or event.ydata is None: return
            x = (2*event.xdata-ncol)/nlig
            y = -(2*event.ydata-nlig)/nlig
            lights[-1].set_position([x,y,0.5])
            render =  shade(normalimage, material, lights)
            render = clip_render(render)
            im.set_data(render)
            plt.draw()

        def onclick(event):
            """
            Create a new light
            """
            if event.xdata is None or event.ydata is None: return
            x = (2*event.xdata-ncol)/nlig
            y = -(2*event.ydata-nlig)/nlig
            lights.append(LightSource([x,y,0.5], light_color(), light_intensity()))
            render =  shade(normalimage, material, lights)
            render = clip_render(render)
            im.set_data(render)
            plt.draw()

        # def onscroll(event):
        #     if event.xdata is None or event.ydata is None: return
        #     x, y, z = lights[-1].position
        #     lights[-1].set_position([x,y,z*(1 + event.step/10)])
        #     render = material.shade(normalimage[:,:,:3],lights)
        #     im.set_data(render)
        #     plt.draw()

        cid = fig.canvas.mpl_connect('motion_notify_event', onmotion)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        # cid = fig.canvas.mpl_connect('scroll_event', onscroll)

        plt.show()

    elif args.mode == "video":
        # Read normal image
        normalimage = plt.imread(imagefile)[:,:,:3]
        mask = normalimage.max(axis=2) <= 4/255
        normalimage = 2 * normalimage - 1
        normalimage = normalimage / np.linalg.norm(normalimage, axis=-1, keepdims=True)
        normalimage[mask] = 0

        files = []
        light = LightSource([-1.,-1.,1.0], light_color(), light_intensity())

        plt.figure(figsize=(15,10))
        for i, x in tqdm(enumerate(np.linspace(-1,1,args.n_frames))):
            y = 1.5*np.sin(2*np.pi*x)
            light.set_position([x,y,1.0])
            render =  shade(normalimage, material, [light])
            render = clip_render(render)
            plt.cla()
            plt.imshow(render)
            fname = "tmp-{}.png".format(i)
            plt.savefig(fname)
            files.append(fname)
        
        subprocess.call("mencoder 'mf://tmp-*.png' -mf type=png:fps={} -ovc lavc "
                    "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg".format(args.fps), shell=True)

        for fname in files:
            os.remove(fname)
    
    else:
        print("Unknown mode.")
    