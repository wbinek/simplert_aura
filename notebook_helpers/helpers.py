import pythreejs as pts
import numpy as np

global camera
camera = None

def display_model(model_display_holder, model3D, simulation):
    #Translate room model to 3d compatible format
    vertices = [v.asArray() for v in model3D._vertices]
    normals = [n.asArray() for n in model3D._normals]
    materials = model3D.materials
    faces = [f._vertices.tolist() + [normals[f.normal_idx], materials[f.mat_name]['kd_hex'], None] for f in model3D._faces]
    viewGeometry = pts.Geometry(vertices=vertices, faces=faces)
    meshModel = pts.Mesh(
        geometry=viewGeometry,
        material=pts.MeshLambertMaterial(vertexColors='VertexColors',
                                             side='FrontSide'))

    #Calculate model centroid
    vert_min = np.min(np.array(vertices), axis=0)
    vert_max = np.max(np.array(vertices), axis=0)
    center = ((vert_min + vert_max)/2).tolist()
    
    # Translate sources to red spheres
    if(simulation.source):
        s_disp_pos = simulation.source.position.asArray()
        dsource = pts.Mesh(geometry=pts.SphereGeometry(),material=pts.MeshLambertMaterial(color='red'),position=s_disp_pos)
    else:
        dsource=None
        
    # Translate receivers to red spheres
    if(simulation.receiver):
        r_disp_pos = simulation.receiver.position.asArray()
        dreceiver = pts.Mesh(geometry=pts.SphereGeometry(),material=pts.MeshLambertMaterial(color='green'),position=r_disp_pos)
    else:
        dreceiver=None
            
    # Generate camera if not avalible
    global camera    
    if not camera:
        key_light = pts.DirectionalLight(color='white', position=[30,2,-15], intensity=0.9)
        camera = pts.PerspectiveCamera(position=[50, 10, -20], up=[0, 1, 0], children=[key_light], aspect=2)
    
    # Build scene from all objects
    scene = pts.Scene(children=[meshModel, dsource, dreceiver, camera, pts.AmbientLight(color='#777777')], background=None)

    # Generate renderer
    renderer = pts.Renderer(camera=camera,
                        scene=scene,
                        alpha=True,
                        clearOpacity=0,
                        width=800,
                        height = 400,
                        controls=[pts.OrbitControls(controlling=camera, target=center)])
    
    #Acquire place for model display and generate image
    model_display_holder.clear_output()
    with model_display_holder:
        display(renderer)