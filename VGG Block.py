def vgg_block(input_layer, n_filters, n_conv):
    for _ in range(n_conv):
        input_layer = Conv2D(n_filters, (3, 3), padding = 'same', activation = 'relu')(input_layer)
    input_layer = MaxPooling2D((2, 2), strides = (2, 2))(input_layer)
    return input_layer
