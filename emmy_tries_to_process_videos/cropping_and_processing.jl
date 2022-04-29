using FileIO, Images, VideoIO, Statistics, DelimitedFiles
# https://juliaio.github.io/VideoIO.jl/stable/reading/

function batchConvertVidToCroppedFrames(path, x₀, x₁, y₀, y₁, Δx, nDigits; invert = false, scaleFactor = 2/3, saveResults=false, saveString = "no_name",skipFrames=0)
    vid = VideoIO.openvideo(path) # open video i/o stream
    VideoIO.seekstart(vid) # reset stream frame/time to zero
    digitsEachFrame = Vector{Matrix{Float64}}[]; # vector to save results in
    v = read(vid) # convert current frame of stream to image variable
    while !eof(vid) # read until end of frame
        read!(vid,v) # advance the stream by one frame and update the image variable
        digits = [imresize(
            Gray.( # convert to grayscale
            v[y₀:y₁,(x₀:x₁) .+ i*Δx] # crop
            ),(30,20) # resize
            ).|> X -> (Int16∘round)(( # round to nearest integer
                    invert ? scaleFactor*(1-Float64(X)) : scaleFactor*Float64(X) # digit features should be ones (white), background should be zeros (black)
                ),digits=0)
        for i in 0:(nDigits-1)] # iterate over digits by translation
        push!(digitsEachFrame,digits) # save to vector

        VideoIO.skipframes(vid,skipFrames,throwEOF=false) # skip frames if you want, by default off
    end

    # convert to a set of ndigits of matrices so that each digit (position in video) has its own matrix
    # the entries of each matrix is the brightness of a pixel whose position (between 20*30 = 600) specified by its column and whose time (between 0 and nFramesTotal) is specified by its row
    matForm = [[reshape(Matrix{Int16}(X[Y]),30*20) for X in digitsEachFrame] for Y in (eachindex∘first)(digitsEachFrame)] .|> X -> (collect∘transpose∘reduce)(hcat,X)
    
    if saveResults # save the data
        for i in eachindex(matForm)
            open(saveString*string(i)*".tsv","w") do io
                writedlm(io,matForm[i])
            end
        end
    end
    return digitsEachFrame, matForm

end

# a utility function for convenience:
numMatRow2Image(matRow) = Gray.(reshape(matRow,(30,20)))