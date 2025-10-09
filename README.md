![output](https://github.com/user-attachments/assets/18f15916-51d3-49d1-846a-d0e19f43e4af)


# CUDA ANIMATED PATH TRACER
*  Eli Asimow
* [LinkedIn](https://www.linkedin.com/in/eli-asimow/), [personal website](https://easimow.com)
* Tested on: Windows 11, AMD Ryzen 7 7435HS @ 2.08GHz 16GB, Nvidia GeForce RTX 4060 GPU


## Overview

This application is an animation path traced renderer for GLTF files. It includes all necessary logic for rendering your custom rigged animation in a dynamic lighting and environment map scene. Models can be diffused textured, or set to one of the supported alternative materials like emission, refraction, or specular. Animations are rendered as separate frames, and can be tied together using video encoders such as ffmpeg. This has been quite a process to create, so I’m excited to share my thoughts with y’all!

## Performance Testing

Let’s look at the effects of our various performance optimizations on renderer’s frames per second. For each of these tests, we compared a set of meshes with increasing complexity, using the Cornell Box scene as a control where appropriate. Unless otherwise specified, the render settings were 8 path bounces, 800x800 pixel screen, with stream compaction and no material sort.

<img width="1920" height="600" alt="Reference (1)" src="https://github.com/user-attachments/assets/7be258db-85b9-44a8-a88b-e29f6c442938" />


### BVH

<img width="600" height="371" alt="Mesh Optimization_ FPS for BVH vs  Naive Triangles" src="https://github.com/user-attachments/assets/f01ef7de-0cf8-4219-a390-539f8b0831d2" />

Starting with something simple, we have the difference between the BVH and Naive Triangles approach to mesh intersection tests. The Naive Triangles algorithm is a simple loop of triangle intersection tests, where the ray checks for collisions against every triangle in the scene. That can get pretty expensive! BVH, on the other hand, optimizes the triangles into a tree hierarchy of bounding boxes, where entire branches of triangles can be eliminated by a simple box collision check. Bounding boxes are optimized along calculated axes to create a balanced binary tree, dropping the time complexity of our procedure here from **O(N)** to **O(log N)**. The performance difference between the two approaches is evident in the data we’ve gathered here. Although Naive triangles manage a respectable level of performance in the simpler crown scene, its performance drops dramatically as triangle count enters the thousands. By Stanford Dragon, the scene fails to render at all! 

### Stream Compaction and Material Sort

Next, let’s look at stream compaction and material sorting. Stream compaction is the process of decreasing our total thread count as paths terminate early in the bounce loop. Material sort is the process of organizing those threads such that paths of shared material type will be processed in shared warps. Our BVH study above showed a fairly expected result. Decreasing the total instruction count resulted in a quicker execution time. The results here, however, buck that trend.

<img width="654" height="403" alt="Memory Optimization Methods Compared" src="https://github.com/user-attachments/assets/6afa0d0f-b09f-49a2-b4c6-96b6a3818a91" />


Woah! Our naive no optimization test case actually outperforms our supposed performance improvements. It’s only in the most complex scene, the Stanford Dragon, that stream compaction begins to outperform no optimizations. Let’s look at material sort first, as that’s the easier case to understand. This path tracer can pull from only four different material types, diffuse, specular, emission, and refraction. With so few material types, it makes sense that the overhead sorting by material type costs more than the time saved with memory adjacency. I’d like to redo these tests in a future edition of this path tracer, where a larger and more complex collection of material types may show this optimization’s worth. 

Much harder to understand is our result for stream compactions. We have relatively little overhead here, just a single thrust partition. How is that costing more than the grand sum of early thread terminations? Let’s take a closer look with a sanity check: how does stream compaction performance change as max bounce increases? If our functionality is correct, this should have relatively little effect on stream compaction, as the vast majority of its threads will bounce into the environment map and terminate before max bounce count is reached. For no optimizations, however, this increase should have a dramatic effect on performance, as all threads will be held hostage for the duration of all bounce count kernels. 

<img width="644" height="398" alt="Stream Compaction Outperforms No Optimizations Over Time" src="https://github.com/user-attachments/assets/20a8df20-d68d-4b72-9d51-cf474e66094c" />

Okay, that’s more like it. Notice how naive’s performance plummets over the course of these increases, while stream compaction’s performance drops seem to be asymptotic. By 48 bounces, stream compaction has begun to outperform its peers, and that gap only increases at 96 bounces and beyond. Okay. So, we know that our stream compaction is functioning properly; why does it have such an overhead performance cost? Let’s boot up NSight Systems and take a closer look.

Here’s the kernel processes and their percent of execution time for no stream compaction:

<img width="1143" height="134" alt="image" src="https://github.com/user-attachments/assets/92c494ab-a7eb-4c4e-86fd-4589906c1b4b" />


And here’s the same for stream compaction:

<img width="1387" height="209" alt="image" src="https://github.com/user-attachments/assets/23d80db3-4425-4b62-81ec-dc785160f509" />


So, the reason for our poor performance is indeed the thrust overhead. At 41.9% of total kernel execution time, our Thrust execution overhead is massive. Anecdotally, I haven’t seen this result recreated in other students’ work. I’d like to explore this more in the future, and determine whether this is a hardware difference or if my thrust partition has gone awry in some way I can’t see. 

## Features ~~~

![output](https://github.com/user-attachments/assets/6a045281-857b-428e-ada0-db93626ef649)


### GLTF Texture, Animation, Joints, Weights, and Mesh Parsing

Any single GLTF file path included in your input json will parse its data into the path trace scene. Textures are managed by a thrust::device_vector, and parsed for color data at the point of ray intersection with a mesh triangle. Worth noting is that I only had time to implement the effects of the base color texture, but all other textures are already parsed and memory copied over to the GPU! When I have some free time I plan to add functionality for normal and roughness maps as well. 

The Mesh parsing is handled in a distinct way to make conversion into bounding volume hierarchies sensible. In an earlier naive implementation, I parsed each mesh primitive separately, and handled their collisions independently. This proved incompatible for BVH, however, so I shifted my approach. Now, all primitive triangle and vertex data are coalesced into two singular buffers that represent the entire scene. This means we can limit ourselves to just one BVH tree, and do one singular BVH stack check that accounts for all meshes at once. 

Lastly, we have the animation functionality. GLTFs are, essentially, a collection of data nodes. In the case of animation, the joints of our actors are represented by joint nodes in that array. Animation channels point to nodes, and represent some transformation of that node over time. These can be rotational, scalar, or translation transformations, and they can be of type scalar, linear, or spline. When joints move, we’ll want to update their corresponding vertices as well. That means we also need to parse out the vertex weight and skin data for each mesh in the scene. My parser gathers all that information, and formats it into a manageable structure under the Scene class. It’s worth noting that animation channels can refer to any kind of node, rather than just joint nodes. Managing transformations of any kind was a bit outside the scope of this project, however, so I’ve restricted the functionality to solely parse joint animation for now. I’d love to expand the scope of what’s supported in the future, although I’ll need to do some more research. 


### Animation

Now that we’ve technically got all the relevant animation information of the GLTF, how do we actually use it? Well, I started by structuring the context of our animation for the path tracer. The first thing we need to do is determine the animation length, which is found by looping over all animation channels and finding the latest end time. Then, we determine how many frames will be rendered by dividing this time by the constant float of 24. That means that all animations will be 24 fps.
Before we can render even frame 0, we need to understand the skin and binding process for mesh based animation. In CG animation, vertices are bound to the joints of their mesh’s skeleton, and as those joints transform, the vertices update in a mirrored manner. Now, this is simple enough when a vertex is only connected to one joint; but what do we do when multiple joints affect the vertex? There, we need weights. In GLTF, vertices can be affected by up to 4 joints. These four weights are represented in a singular vec4, where their sum always adds to 1. A separate vec4 of joints indices represents the joints with which these weights correspond. Each vertex has one weight vec4, and one joint indices vec4.

With that context covered, the process for animation is actually just three fairly simple steps. The first step is to parse the animation movement. For each animation channel, we find the joint node its transformation corresponds to, and we update that joint’s position, transformation, and scale to its next frame value. These updates are to the joint’s local transformation matrix, local to its parent frame. And so, the second step is to determine each joint’s global transformation matrix, determining their position, scale and rotation, not relative to their parent, but relative to the world. Lastly, the third step is to apply these new joint positions to the vertices. Each vertex uses the product of their four connected joints’ global matrices and skin inverse bind matrices to determine their new position as a weighted average. We rebuffer that new vertex data to the GPU, and we have our animated mesh!

In terms of how the path tracer handles the animation, on completion of our current frame we simply save our current render, clear the screen, update the mesh, and begin our next frame. When all frames have been animated, the render is complete and the application terminates. I use ffmpeg outside of my application to combine the render pngs into videos and gifs. 


### OPTIX Denoiser

	Rendering all the frames of an animation takes quite a bit longer than rendering a singular image. When I briefly discussed my plan with Shehzan, his advice was to include a denoiser to expedite that process. Hooking up the denoiser was actually fairly straightforward after I had downloaded the Optix headers. The Beauty texture we already had, as we’d only want to apply the denoiser at the end of the render, and adding a scene normals render was fairly straightforward. Honestly, I should have added the normal render earlier; it caught a few nasty bugs with my animation normals. 


The results were immediately compelling, but they had their constraints in an animation context. Consider the difference between the renders of two adjacent frames; we’ll have the updated model, but we’ll also have a slightly different environment as the path rays diverge to slightly different results. This means we’ll have fairly noticeable denoiser artifacts in our animation as Optix reacts differently to the slight variance of our background pixels. 
Optix actually offers a solution for this, with temporal rendering. We can include the previous frame’s denoised result as context for our current frame! Unfortunately, I just couldn’t get this element of the denoiser to work. I would pass in the data, and the final denoised result would fail, outputting a black screen. Definitely something to return to in the future. Still, the denoise functionality that I achieved here is worth using, and helped me render many of these animations in under 30 minutes.

### Environment Map

Environment map was a fairly easy feature to add, although its usage is pretty limited by the current rendering approach. We simply read in an HDR file as a Cuda texture, and any ray that ‘escapes’ the scene multiplies its color by a sampled point in that texture. We use the direction of the ray’s vector to determine its uv position. I really love how this adds to the renders, but I was frustrated by the sprinkles of noise it results in. If a light ray bounces off an object and happens to hit an uncommon bright spot in the environment, it will greatly affect that path’s average color. 


In cases like this, the underlying color of the object is so greatly affected that even denoising can’t save the render.



This could be solved by including Multiple Importance Sampling in my render pipeline, or perhaps a gaussian blur of the sampled texture to normalize its colors. Definitely something to look into in the future.


## Closing Thoughts

Considering the timeline of ~two weeks of development, I’m happy with the results here. The night I completed my first render, the dancer on the lake, I must have stayed up for an hour watching and rewatching the video. The whole experience was pretty damn cool. That doesn’t mean the work’s done, though. There were so many moments that came up while writing this readme that I thought of features that would greatly improve the work, things I could accomplish in just a few days. I’m excited to return to path tracing soon. I’ll leave you with some funny bloopers from my trials and tribulations in development. 
