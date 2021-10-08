def rotate(axis, angle, system: Opts = Opts()) -> SimpleNamespace:


""" align set alignment of the objects in the block

   [...] = rotate(axis, angle, obj <, obj> ...);

   Rotates the corresponding gradinet object(s) about the given axis by 
   the specified amount. Gradients parallel to the rotation axis and 
   non-gradient objects are not affected. 
   Possible rotation axes are 'x', 'y' or 'z'.

   Returns either a cell-array of objects if one return parameter is
   provided or an explicit list of objects if multiple parameters are
   given. Can be used directly as a parameter of seq.addBlock().
"""
axes = ['x', 'y', 'z']
# cycle through the objects and rotate gradients non-parallel to the given rotation axis
# Rotated gradients assigned to the same axis are then added together
# First create indexes of the objects to be bypassed or rotated
irotate1 = []
irotate2 = []
ibypass = []

# axes2rot=axes(~strcmp(axes,axis))
# if length(axes2rot)!=2:
#   error('incorrect axis specification')
