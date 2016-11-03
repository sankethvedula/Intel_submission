require "image"
require "cunn"
require "nn"

local function rip2pieces(image1,image2,mlp)

	local height = image1:size(1)
	local width  = image2:size(2)

	local recon_image = torch.zeros(height, width)

	for i = 1, height-16,1 do
		--print(i)
		for j = 1, width-16,1 do
			patch_1  = image1[{ {i, i+15}, {j,j+15} }]
			patch_2  = image2[{ {i, i+15}, {j,j+15} }]

			test_patch_1 = patch_1:clone() -- Don't forget to clone()
			test_patch_2 = patch_2:clone()


			output = mlp:forward(torch.cat(
					patch_1:reshape(1,256),
					patch_2:reshape(1,256),
					2):cuda())


			output = output:reshape(8,8):double()
			test_output = output:clone()

			image_out = test_output:add(1):mul(255./2.):byte()

			temp =	torch.add(recon_image[{ {i+4,i+11}, {j+4,j+11} }],output)

			recon_image[{ {i+4,i+11},{j+4,j+11} }]:copy(temp)

   		patch_image_1 = test_patch_1:add(1):mul(255./2.):byte()
			patch_image_2 = test_patch_2:add(1):mul(255./2.):byte()

		end
	end


	print(recon_image:size())
	out_image = recon_image:div(64):add(1):mul(255./2.):byte()
	return out_image

end

number_of_test = 1000
count = 0
for file = 1, 1000 do
	print(file)
	local mlp = torch.load("nets/mlp_adam.t7")

		image1 = image.load("./test/"..file..".png",1,'byte')
		image2 = image.load("./test/"..(file+1)..".png",1,'byte')
		image1 = image.scale(image1,400,400)
		image2 = image.scale(image2,400,400)
		local image1 = image1:double():mul(2./255.):add(-1)
		local image2 = image2:double():mul(2./255.):add(-1)

		out_image = rip2pieces(image1,image2,mlp)

		image.save("./adam_out/"..count..".png",image1:add(1):mul(255./2.):byte())
		image.save("./adam_out/"..(count+1)..".png",out_image)
		image.save("./adam_out/"..(count+2)..".png",image2:add(1):mul(255./2.):byte())
		count = count+2
	--rip2pieces(mlp)
end
