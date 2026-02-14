#=
A material is a description of the appearence of an object.

There are numerous models for a material (Phong, Blinn–Phong, Cook–Torrance,
Lambert, Oren–Nayar, Minnaert, etc), covering wildly different materials such
as glass, metal, rubber, clay, etc. Unfortunately, different render engines
implement different subsets of those models, making it impossible to have portable
code that depends on a specific model. Additionally, each of these models have
different parameters, some of which might be difficult to specify by a non-expert.
Therefore, we prefer to provide a more intuitive approach, based on providing
different categories of materials, each with a minimal number of parameters.
Each backend is responsible for selecting the most adequate reflection model and
convert the generic material parameters into specific model parameters.

Note: eta is the index of refraction (IOR)
=#


#=

We will base this unified model in Filament's standard model
https://github.com/google/filament
whose documentation I shamelessly copy below:

The appearance of a material using the standard model is controlled using the
properties described in table [standardProperties].

        Property        |      Definition
-----------------------:|:---------------------
**baseColor**           | Diffuse albedo for non-metallic surfaces, and specular color for metallic surfaces
**metallic**            | Whether a surface appears to be dielectric (0.0) or conductor (1.0). Often used as a binary value (0 or 1)
**roughness**           | Perceived smoothness (1.0) or roughness (0.0) of a surface. Smooth surfaces exhibit sharp reflections
**reflectance**         | Fresnel reflectance at normal incidence for dielectric surfaces. This directly controls the strength of the reflections
**sheenColor**          | Strength of the sheen layer
**sheenRoughness**      | Perceived smoothness or roughness of the sheen layer
**clearCoat**           | Strength of the clear coat layer
**clearCoatRoughness**  | Perceived smoothness or roughness of the clear coat layer
**anisotropy**          | Amount of anisotropy in either the tangent or bitangent direction
**anisotropyDirection** | Local surface direction
**ambientOcclusion**    | Defines how much of the ambient light is accessible to a surface point. It is a per-pixel shadowing factor between 0.0 and 1.0
**normal**              | A detail normal used to perturb the surface using _bump mapping_ (_normal mapping_)
**bentNormal**          | A normal pointing in the average unoccluded direction. Can be used to improve indirect lighting quality
**clearCoatNormal**     | A detail normal used to perturb the clear coat layer using _bump mapping_ (_normal mapping_)
**emissive**            | Additional diffuse albedo to simulate emissive surfaces (such as neons, etc.) This property is mostly useful in an HDR pipeline with a bloom pass
**postLightingColor**   | Additional color that can be blended with the result of the lighting computations. See `postLightingBlending`
**ior**                 | Index of refraction, either for refractive objects or as an alternative to reflectance
**transmission**        | Defines how much of the diffuse light of a dielectric is transmitted through the object, in other words this defines how transparent an object is
**absorption**          | Absorption factor for refractive objects
**microThickness**      | Thickness of the thin layer of refractive objects
**thickness**           | Thickness of the solid volume of refractive objects
[Table [standardProperties]: Properties of the standard model]

The type and range of each property is described in table [standardPropertiesTypes].

        Property        |   Type   |            Range         |           Note
-----------------------:|:--------:|:------------------------:|:-------------------------
**baseColor**           | float4   |  [0..1]                  | Pre-multiplied linear RGB
**metallic**            | float    |  [0..1]                  | Should be 0 or 1
**roughness**           | float    |  [0..1]                  |
**reflectance**         | float    |  [0..1]                  | Prefer values > 0.35
**sheenColor**          | float3   |  [0..1]                  | Linear RGB
**sheenRoughness**      | float    |  [0..1]                  |
**clearCoat**           | float    |  [0..1]                  | Should be 0 or 1
**clearCoatRoughness**  | float    |  [0..1]                  |
**anisotropy**          | float    |  [-1..1]                 | Anisotropy is in the tangent direction when this value is positive
**anisotropyDirection** | float3   |  [0..1]                  | Linear RGB, encodes a direction vector in tangent space
**ambientOcclusion**    | float    |  [0..1]                  |
**normal**              | float3   |  [0..1]                  | Linear RGB, encodes a direction vector in tangent space
**bentNormal**          | float3   |  [0..1]                  | Linear RGB, encodes a direction vector in tangent space
**clearCoatNormal**     | float3   |  [0..1]                  | Linear RGB, encodes a direction vector in tangent space
**emissive**            | float4   |  rgb=[0..n], a=[0..1]    | Linear RGB intensity in nits, alpha encodes the exposure weight
**postLightingColor**   | float4   |  [0..1]                  | Pre-multiplied linear RGB
**ior**                 | float    |  [1..n]                  | Optional, usually deduced from the reflectance
**transmission**        | float    |  [0..1]                  |
**absorption**          | float3   |  [0..n]                  |
**microThickness**      | float    |  [0..n]                  |
**thickness**           | float    |  [0..n]                  |
[Table [standardPropertiesTypes]: Range and type of the standard model's properties]

The @defproxy(standard_material, ...) is defined in Shapes.jl (before the pre-defined
material constants that depend on it). The realize method auto-generated by @defproxy
calls b_standard_material, defined below.
=#

# Default b_standard_material delegates to b_new_material, forwarding
# all Filament PBR parameters.
export b_standard_material
b_standard_material(b::Backend, name, layer, base_color, metallic, roughness,
                    reflectance, sheen_color, sheen_roughness,
                    clear_coat, clear_coat_roughness,
                    anisotropy, anisotropy_direction,
                    ambient_occlusion, normal, bent_normal, clear_coat_normal,
                    emissive, post_lighting_color,
                    ior, transmission, absorption, micro_thickness, thickness) =
  b_new_material(b, name, base_color, metallic, reflectance, roughness,
                 clear_coat, clear_coat_roughness, ior,
                 transmission, roughness,
                 emissive, 1.0,
                 sheen_color, sheen_roughness,
                 anisotropy, anisotropy_direction,
                 ambient_occlusion, normal, bent_normal, clear_coat_normal,
                 post_lighting_color,
                 absorption, micro_thickness, thickness)
