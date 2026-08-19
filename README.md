# io_scene_glacier
A Blender 3.X/4.X+ **work-in-progress** plugin to import and export assets for IO Interactive's iterations of the Glacier Engine, Expect it to be updated in the near future.
***(THIS PLUGIN WAS TESTED WITH STEAM VERSIONS OF THE GAMES, OTHER VERSIONS ARE UNTESTED!)***

# Features
- **Global**: Functions and parsers are well-documented and print information in the System Console. *(In case you're curious or troubleshooting)*
- **Models**: A "Mesh Properties" Menu to define mesh data such as type, general properties and LOD data.
- **Models**: An import option for the model's collision / hitbox data to be built into the scene. *(Required for exporting!)*
- **Models**: An import option to import the game's original normals or to recalculate them. *(Looks smoother)*
- **Models**: The ability to switch vertex groups' bone indexes/indices to bone names and vice versa. *(The latter is required for exporting!)*
- **Models**: The ability to assign random colors to materials or not. *(To help with distinguishing submeshes)*
- **Skeletons**: X-Axis Mirroring is supported.
- **Textures**: Bulk import is supported.
- **Textures**: DirectXTex' texconv is used for converting formats unsupported by Blender and for exporting custom textures.

# Glacier1 File Support
*010 Editor Binary Templates for the file formats can be found [here](https://github.com/TheDodylectableX/Research/010%20Editor/tree/main/IO%20Interactive%20Glacier1%20(PC)) on my personal repository*
| Asset Type              | File Type  | Import Support | Export Support |
| ----------------------- | ---------- | -------------- | -------------- |
| Global Mission Script   | .GMS       | Yes            | No             |
| RenderPrimitive Archive | .PRM       | Yes            | No             |
| RenderTexture Archive   | .TEX       | Yes            | No             |
### Supported Games
- Hitman 2: Silent Assassin
- Freedom Fighters
- Hitman: Contracts
- Hitman: Blood Money and Mini Ninjas
# Glacier2 File Support
*010 Editor Binary Templates for the file formats can be found [here](https://github.com/TheDodylectableX/Research/010%20Editor/tree/main/IO%20Interactive%20Glacier2%20(PC)) on my personal repository*
| Asset Type              | File Type  | Import Support | Export Support |
| ----------------------- | ---------- | -------------- | -------------- |
| RenderPrimitive         | .PRIM       | Yes           | WIP            |
| RenderTexture           | .TEXT/.TEXD | Yes           | WIP            |
| BoneRig                 | .BORG       | Yes           | WIP            |
| CloakWorks Shroud Cloth | .CLOS       | Yes           | WIP            |
### Supported Games
- Hitman: Absolution
- Hitman: World of Assassination
- 007 First Light

# Notes
- My reverse engineering of Glacier2 is way ahead in terms of progress compared to the predecessor so RenderPrimitive importing for example will have issues.
- G1: .PRM/.TEX are archives that consist of many models and textures inside of them and not separate as on G2.
- G2: .TEXT files only carry internal lower quality mips from WoA onwards, The importer will ask you to point out to the higher quality mip file which is .TEXD.

# Credits
- Glacier2 Modding Community (Pavle, Dafitius, PawREP, jrdeveraux and more): Massive help in understanding Glacier2 and it's file formats
- id-daemon: Old tools which helped in understanding Glacier1 and it's file formats
