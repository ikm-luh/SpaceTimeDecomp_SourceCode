
using SparseArrays
using JuAFEM

"""
Reads Abaqus-Mesh from *.inp file (supported as export format in GMSH)
    input: file (path + name)
    output: grid, cell_type, nodes_by_volumes, nodes_by_surfaces, cells_by_volumes
        grid::Grid -> complete 'Grid' data for JuAFEM incl 'boundary_matrix' (with emtpy face- and cell-sets)
        cell_type::AbstractCell -> reader is only suitable unmixed meshes (only one cell type)
        nodes_by_volumes[1][:]: String-List of volume-ELSET; nodes_by_volumes[2][k]: Integer-List with nodes in ELSET with name == nodes_by_volumes[1][k]
        nodes_by_surfaces[1][:]: String-List of suface-ELSET; nodes_by_surfaces[2][k]: Integer-List with nodes in ELSET with name == nodes_by_surfaces[1][k]
        cells_by_volumes[1][:]: String-List of volume-ELSET; cells_by_volumes[2][k]: Integer-List with cells in ELSET with name == cells_by_volumes[1][k]
"""
function mesh_from_inp(fname::String)

    supported_ele_types_3D = ["C3D4","C3D8","C3D10"]
    supported_ele_types_2D = ["CPS3","CPS4","CPS6"]

    # get data from file
    f = open(fname,"r")
    println("  reading mesh file ",fname)
    data = read(f, String)

    # get element types and number of spacial dimensions
    nsd = 0
    found_types = String[]
    typeposis = findall("type=",data)
    for typeposi in typeposis
        typeposi_start = typeposi[end]+1
        typeposi_end = findfirst(',',data[typeposi_start:end]) + typeposi_start - 2
        unique!(push!(found_types,data[typeposi_start:typeposi_end]))
    end

    # check if element_type is supported
    # check 3D
    ele_type = intersect(found_types,supported_ele_types_3D)
    if !isempty(ele_type)
        nsd = 3
    else
        # check 2D
        ele_type = intersect(found_types,supported_ele_types_2D)
        if !isempty(ele_type)
            nsd = 2
        else
            error("No supported element types found!")
        end
    end

    # define according surface type
    if length(ele_type) > 1
        error("Mixed meshes are not supported!")
    else
        ele_type = ele_type[1]
    end
    cell_type = nothing
    surf_type = ""
    if nsd == 3
        if ele_type == "C3D4"
            surf_type = "CPS3"
            cell_type = JuAFEM.Tetrahedron
        elseif ele_type == "C3D8"
            surf_type = "CPS4"
            cell_type = JuAFEM.Hexahedron
        elseif ele_type == "C3D10"
            surf_type = "CPS6"
            cell_type = JuAFEM.QuadraticTetrahedron
        else
            error("Non-supported element type ''",ele_type,"'!")
        end
    else # nsd ==2
        if ele_type == "CPS3"
            surf_type = "T3D2"
            cell_type = JuAFEM.Triangle
        elseif ele_type == "CPS4"
            surf_type = "T3D2"
            cell_type = JuAFEM.Quadrilateral
        elseif ele_type == "CPS6"
            surf_type = "T3D3"
            cell_type = JuAFEM.QuadraticTriangle
        else
            error("Non-supported element type ''",ele_type,"'!")
        end
    end
    close(f)

    # get mesh data
    f = open(fname,"r")
    nodes = []
    cells = []
    cells_ELSET = []
    surfaces = []
    surfaces_ELSET = []
    active = ""
    set_name = ""


    while !eof(f) # go line-wise through whole file
        # find header
        line = strip(readline(f))
        if line[1] == '*' # header lines start with '*'
            if occursin("*NODE",line)
                active = "node"
            elseif occursin(ele_type,line)
                active = "cell"
                set_name = line[findlast('=',line)+1:end]
            elseif occursin(surf_type,line)
                active = "surface"
                set_name = line[findlast('=',line)+1:end]
            else
                active = ""
            end

        else # data lines start without '*'
            line_data = strip.(split(line,',',keepempty=false))
            if active == "node"
                push!(nodes,parse.(Float64, line_data)[2:nsd+1])
            elseif active == "cell"
                push!(cells,parse.(Int, line_data)[2:end])
                push!(cells_ELSET,set_name)
            elseif active == "surface"
                new_face = parse.(Int, line_data)[2:end]
                counter_face = findall(x -> issetequal(x,new_face),surfaces)
                if isempty(counter_face) # only add unique faces
                    push!(surfaces,new_face)
                    push!(surfaces_ELSET,set_name)
                else # delete face if counter_face was found
                    deleteat!(surfaces,counter_face)
                    deleteat!(surfaces_ELSET,counter_face)
               end
            end
        end
    end
    close(f)

    # find loose nodes
    nnodes = size(nodes,1)
    ncells = size(cells,1)
    println("  found $ele_type-mesh with $ncells cells and $nnodes nodes...")
    adj_cells = get_con_cells(nnodes,ncells,cells)
    loose_nodes = Integer[]
    for (node,cells) in enumerate(adj_cells)
        if isempty(cells)
            push!(loose_nodes,node)
        end
    end
    sort!(loose_nodes)

    # removing loose nodes (i.e. not connected to any cell) from data
    if isempty(loose_nodes)
        println("  no loose nodes detected.")
    else
        n_loose_nodes = length(loose_nodes)
        if n_loose_nodes > 10
            println("  deleting ",n_loose_nodes," loose nodes...")
        else
            println("  deleting loose nodes ",string(loose_nodes))
        end
        # delete coordinates
        deleteat!(nodes,loose_nodes)
        # reduce node numbers in cells and surfaces
        for h_node in reverse(loose_nodes)
            for cell_nodes in cells
                hits = findall(x -> x > h_node,cell_nodes)
                cell_nodes[hits] = cell_nodes[hits] .- 1
            end
            for surf_nodes in surfaces
                hits = findall(x -> x > h_node,surf_nodes)
                surf_nodes[hits] = surf_nodes[hits] .- 1
            end
        end
    end

    # convert to JuAFEM and construct set_grid
    nnodes = size(nodes,1)
    ncells = size(cells,1)
    println("  imported $ele_type-mesh with $ncells cells and $nnodes nodes...")
    j_nodes = Vector{JuAFEM.Node{nsd,Float64}}(undef,nnodes)
    j_cells = Vector{cell_type}(undef,ncells)
    Threads.@threads for i = 1:nnodes
        j_nodes[i] = JuAFEM.Node(Vec{nsd}(nodes[i]))
    end
    Threads.@threads for i = 1:ncells
        j_cells[i] = cell_type(Tuple(cells[i]))
    end

    # number of faces per cell
    nfc = 0
    if cell_type == Quadrilateral
        nfc = 4
    elseif cell_type == Triangle || cell_type == QuadraticTriangle
        nfc = 3
    elseif cell_type == Hexahedron
        nfc = 6
    elseif cell_type == Tetrahedron || cell_type == QuadraticTetrahedron
        nfc = 4
    end

    # faces on boundary
    neigh_cells = get_neighbor_cells(nnodes,ncells,cells)
    nth = Threads.nthreads()
    boun_mat = [ones(Bool,nfc,ncells) for i=1:nth] # one for each thread, ones = true

    progress = 0
    Threads.@threads for pivot_ele = 1:ncells
        th = Threads.threadid()
        if th == 1
            progress_new = Integer(round(nth*100.0*pivot_ele/ncells,digits=0))
            if progress_new > progress
                print("\r  computing boundary matrix...$progress%")
                progress = progress_new
            end
        end
        pivot_faces = JuAFEM.faces(j_cells[pivot_ele])
        for (pf,pivot_face) in enumerate(pivot_faces) # loop thorough pivot element's faces
            if boun_mat[th][pf,pivot_ele]
                found = false
                for neigh_ele in neigh_cells[pivot_ele] # Loop through neighbor elements
                    neigh_faces = JuAFEM.faces(j_cells[neigh_ele])
                    for (nf,neigh_face) in enumerate(neigh_faces) # loop thorough neighbor element's faces
                        if issetequal(pivot_face,neigh_face) & boun_mat[th][nf,neigh_ele]
                            boun_mat[th][pf,pivot_ele] = false
                            boun_mat[th][nf,neigh_ele] = false
                            found = true
                        end
                        found ? break : nothing
                    end
                    found ? break : nothing
                end
            end
        end
    end
    print("\r  computing boundary matrix...100%")
    # add all threades boun_mat up
    boun_mat_final = boun_mat[nth]
    for th = 1:nth-1
        boun_mat_final = boun_mat_final .* boun_mat[th]
    end

    if sum(boun_mat_final) == false # not a single entry
        @warn("No boundaries found!")
    end

    println("\ndone reading mesh file!")

    # nodes sorted by "ELSET"
    cells_ELSET_names = unique(cells_ELSET)
    cells_by_ELSET = [Int[] for i=1:length(cells_ELSET_names)]
    for (i,set_name) in enumerate(cells_ELSET)
        j = findfirst(x-> x==set_name,cells_ELSET_names)
        push!(cells_by_ELSET[j],i)
    end
    cells_ELSET_nodes = [Int[] for i=1:length(cells_ELSET_names)]
    for (i,nodes) in enumerate(cells)
        j = findfirst(x-> x==cells_ELSET[i],cells_ELSET_names)
        cells_ELSET_nodes[j] = union!(cells_ELSET_nodes[j],nodes)
    end
    surfaces_ELSET_names = unique(surfaces_ELSET)
    surfaces_ELSET_nodes =[Int[] for i=1:length(surfaces_ELSET_names)]
    for (i,nodes) in enumerate(surfaces)
        j = findfirst(x-> x==surfaces_ELSET[i],surfaces_ELSET_names)
        surfaces_ELSET_nodes[j] = union!(surfaces_ELSET_nodes[j],nodes)
    end

    grid = JuAFEM.Grid(j_cells, j_nodes, boundary_matrix=sparse(boun_mat_final))
    nodes_by_volumes = (cells_ELSET_names,sort!.(cells_ELSET_nodes))
    nodes_by_surfaces = (surfaces_ELSET_names,sort!.(surfaces_ELSET_nodes))
    cells_by_volumes = (cells_ELSET_names,sort!.(cells_by_ELSET))


    return grid, cell_type, nodes_by_volumes, nodes_by_surfaces, cells_by_volumes

end

"""
Returns con_cells = Vector{Vector::Integer}: con_cells[n] = Vector with cells connected to node "n"
"""
function get_con_cells(nn::Integer,ne::Integer,cells)
    con_cells = [Integer[] for i = 1:nn]
    # filling with data
    for ele = 1:ne # Loop through all elements
        lNod = cells[ele] # local nodes
        for i = 1:length(lNod)                  # loop through all local nodes
            push!(con_cells[lNod[i]],ele)
        end
    end
    return con_cells
end

"""
Returns neigh_cells = Vector{Vector::Integer}: neigh_cells[n] = Vector with direct neighbor cells of cell "n"
"""
function get_neighbor_cells(con_cells::Vector{Vector{Integer}},ne::Integer,cells)
    neigh_cells = [Integer[] for i = 1:ne]
    # filling with data
    for ele = 1:ne # Loop through all elements
        lNod = cells[ele] # local nodes
        for node in lNod # loop through all local nodes
            union!(neigh_cells[ele],con_cells[node])
        end
        filter!(e->e≠ele,neigh_cells[ele]) # remove element itself from list
    end
    return neigh_cells
end
@inline get_neighbor_cells(nn::Integer,ne::Integer,cells) = get_neighbor_cells(get_con_cells(nn,ne,cells),ne,cells)


###################################################################################################################

###################################################################################################################


"""
Saves *.inp with surface elements from file 'fname' in basis plane with normal-direction 'dir' and base coordinate 'coord'.
Important remark:
    -> 'dir' is sign sensitive -> normal should point away from volume
"""
function extract_surface_from_inp(fname::String,dir::Integer,coord::Float64,fname_out::String)
    gap_tol = 1.0e-15
    # get mesh data
    f = open(fname,"r")
    println("  extracting surface from mesh file ",fname,"...")
    data = read(f, String)
    # get element type
    surf = findfirst("ELSET=Surface",data)
    posi1 = findlast('=',data[1:surf[1]])
    posi2 = findlast(',',data[1:surf[1]])
    ele_type = data[posi1+1:posi2-1]

    if ele_type == "CPS3"
        surf_type = "T3D2"
        cell_type = JuAFEM.Triangle
    elseif ele_type == "CPS4"
        surf_type = "T3D2"
        cell_type = JuAFEM.Quadrilateral
    elseif ele_type == "CPS6"
        surf_type = "T3D3"
        cell_type = JuAFEM.QuadraticTriangle
    else
        error("Non-supported element type ''",ele_type,"'!")
    end
    close(f)

    # prepare data fields
    nodes = []
    cells = []
    surfaces = []
    new_node_numbers = []
    active = ""
    set_name = ""
    nod_count = 0

    f = open(fname,"r")
    while !eof(f) # go line-wise through whole file
        # find header
        line = strip(readline(f))
        if line[1] == '*' # header lines start with '*'
            if occursin("*NODE",line)
                active = "node"
                push!(nodes,line)
            elseif occursin(surf_type,line)
                active = "surface"
                set_name = line[findlast('=',line)+1:end]
                push!(surfaces,"*ELEMENT, type="*surf_type*", ELSET="*set_name)
            elseif occursin(ele_type,line)
                active = "cell"
                set_name = line[findlast('=',line)+1:end]
                push!(cells,"*ELEMENT, type="*ele_type*", ELSET="*set_name)
            else
                active = ""
            end
        else # data lines start without '*'
            line_data = strip.(split(line,',',keepempty=false))
            if active == "node"
                coordinates = parse.(Float64, line_data[2:4])
                if abs(coordinates[abs(dir)] - coord) ≤ gap_tol # take node
                    nod_count = nod_count + 1
                    if abs(dir) == 1  # resorting
                        coordinates_2D = coordinates[[2,3,1]]
                    elseif  abs(dir) == 2
                        coordinates_2D = coordinates[[3,1,2]]
                    elseif  abs(dir) == 3
                        coordinates_2D = coordinates[[1,2,3]]
                    end
                    push!(nodes,coordinates_2D)
                    push!(new_node_numbers,nod_count)
                else # drop node
                    push!(new_node_numbers,0)
                end
            elseif active ∈ ["surface","cell"]
                s_nodes = parse.(Int, line_data)[2:end]
                new_s_nodes = zeros(Int,size(s_nodes))
                new_s_nodes[:] .= new_node_numbers[s_nodes[:]]
                if 0 ∉ new_s_nodes # ignored nodes have new id == 0
                    dir < 0 ? reverse!(new_s_nodes) : nothing
                    if active == "surface"
                        push!(surfaces,new_s_nodes)
                    elseif active == "cell"
                        push!(cells,new_s_nodes)
                    end
                end
            end
        end
    end
    close(f)

    # delete empty surfacesets
    delete = Int[]
    for i = 1:length(surfaces)-1
        surfaces[i][1] == '*' && surfaces[i+1][1] == '*' ? push!(delete,i) : nothing
    end
    surfaces[end][1] == '*' ? push!(delete,length(surfaces)) : nothing
    deleteat!(surfaces,delete)

    # delete empty cellsets
    delete = Int[]
    for i = 1:length(cells)-1
        cells[i][1] == '*' && cells[i+1][1] == '*' ? push!(delete,i) : nothing
    end
    cells[end][1] == '*' ? push!(delete,length(cells)) : nothing
    deleteat!(cells,delete)

    # writing data to file
    f_out = fname_out
    if f_out == ""
        f_out = fname[1:end-4]*"_2D.inp"
    end
    io = open(f_out, "w")
    println("  writing 2D-mesh file ",f_out,"...")
    println(io,"*Heading")
    println(io," 2D-Surface-Mesh of "*fname*" in plane i = $dir with x_i = $coord")
    println(io,nodes[1])
    for (i,coord) in enumerate(nodes[2:end])
        println(io,"$i, $(coord[1]), $(coord[2]), $(coord[3])")
    end
    println(io,"******* E L E M E N T S *************")
    ele_count = 0
    for line in surfaces
        if typeof(line) == Array{Int64,1}
            ele_count = ele_count + 1
            print(io,ele_count)
            for node in line
                print(io,", $node")
            end
            print(io,"\n")
        else
            println(io,line)
        end
    end
    for line in cells
        if typeof(line) == Array{Int64,1}
            ele_count = ele_count + 1
            print(io,ele_count)
            for node in line
                print(io,", $node")
            end
            print(io,"\n")
        else
            println(io,line)
        end
    end
    flush(io)
    close(io)
    #   - write to file: "*Heading","surfaces $dir at $coord from file...", nodes, surfaces, cells
    #   - implement in 'inp_reader.jl'
    #   - test it
    println("  ... surface extraction done!")
    nnodes = length(nodes)-1
    println("  2D-Mesh with $nnodes nodes exported to "*f_out)
    #return nodes,surfaces,cells # only for testing
    return nothing

end
@inline extract_surface_from_inp(fname::String,dir::Integer,coord::Float64) = extract_surface_from_inp(fname,dir,coord::Float64,"")
@inline extract_surface_from_inp(fname::String,fname_out::String) = extract_surface_from_inp(fname,-3,0.0,fname_out)
@inline extract_surface_from_inp(fname::String) = extract_surface_from_inp(fname,-3,0.0,"")


"""
Produces Paraview file with cell coloring accoring to volume_cells (last output from "mesh_from_inp").
Indicates belonging elements with value 1 (and non belonging elements with 0) or plots the set_index of the position in the list.
"""
function plot_cell_sets(FileName::String,grid,volume_cells)
    vtk_grid(FileName,grid) do vtk
        set_indices = zeros(length(grid.cells))
        for (i,fieldname) in enumerate(volume_cells[1])
            data = zeros(length(grid.cells))
            data[volume_cells[2][i]] .= 1.0
            vtk_cell_data(vtk, data, fieldname)
            set_indices[volume_cells[2][i]] .= i
        end
        vtk_cell_data(vtk, set_indices, "set_index")
    end
end

"""
Returns a Integer Vector with set index for each cell accoring to ordering  in "volume_cells".
    Optional input: replace = Vector{Integer or Float64} with length equal to number of volume sets: replaces cell index by specified value in the given vector
        e.g.: cellwise_setindex(grid,volume_cells, replace=[1,1,2]) returns value 1 for cells in the first end second volume set and value 2 for cells in the third set;
    Optional input: plot = Sting: plots cell indeces to ParaView file with given filepath/name
"""
function cellwise_set_index(grid,volume_cells; replace=Integer[], plot="")
    n_sets = length(volume_cells[1])
    if isempty(replace)
        replace = [i for i= 1:n_sets]
    else
        if length(replace) ≠ n_sets
            if length(replace) ≠ 1
                error("Length of 'replace' do not match number of cell sets!") : nothing
            else
                replace = [replace[1] for i= 1:n_sets] # all get the same index
            end
        end
    end

    cell_indices = zeros(length(grid.cells))
    for i = 1:n_sets
        cell_indices[volume_cells[2][i]] .= replace[i]
    end
    if !isempty(plot)
        vtk_grid(plot,grid) do vtk
            vtk_cell_data(vtk, cell_indices, "set_index")
        end
    end
    return cell_indices
end

"""
Returns a String Vector with set name for each cell according to "cells_by_volumes" (last output from "mesh_from_inp").
"""
function cellwise_name_list(grid,cells_by_volumes)
    Cellindex = cellwise_set_index(grid,cells_by_volumes)
    return [cells_by_volumes[1][Integer(Cellindex[i])] for i=1:length(Cellindex)]
end
