--require "cunn"
require "nn"
require "image"
require "optim"

train_data = torch.load("validation.t7")
patch_size = 16
number_of_images = 1000

table_of_patches = {}


local function create_patches(train_data,patch_size,number_of_images)
  for num = 1, number_of_images do
    print(num)
    image_1 = train_data[num]

    height = image_1:size(1)
    width = image_1:size(2)
    number_of_patches = ((height*width)/(16*16))

    tensor_of_images = torch.zeros(number_of_patches,16,16)

    --print(number_of_patches)
    count = 0
    for i = 1,height-16,16 do
      for j = 1,width-16,16 do
        single_patch = image.crop(image_1, j,i,j+16,i+16)
        count = count+1
        tensor_of_images[{ count,{},{} }] = single_patch
      end
    end
    table.insert(table_of_patches,tensor_of_images)
  end
  return table_of_patches
end

table_of_patches = create_patches(train_data,patch_size,number_of_images)
torch.save("validation_table_of_patches.t7",table_of_patches)
