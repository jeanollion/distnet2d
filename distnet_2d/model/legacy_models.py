
#####################################################################
#################### LEGACY VERSION #################################
#####################################################################



# one encoder per input + one decoder + one last level of decoder per output + custom frame window size
def get_distnet_2d_sep_out_fw(input_shape, # Y, X
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride, maxpool_and_stride
            combine_kernel_size:int = 3,
            skip_stop_gradient:bool = False,
            predict_contours:bool = False,
            encoder_settings:list = None,
            feature_settings: list = None,
            decoder_settings: list = None,
            decoder_center_settings : list = None,
            attention : bool = True,
            self_attention:bool = True,
            residual_combine_size:int = 3,
            frame_window:int = 1,
            predict_center = False,
            edm_center_mode = "MEAN", # among MAX, MEAN, SKELETON or NONE
            next:bool=True,
            name: str="DiSTNet2D",
            l2_reg: float=1e-5,
    ):
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"
        if predict_center:
            assert decoder_center_settings is not None, "decoder_center_settings cannot be none"
            assert len(encoder_settings)==len(decoder_center_settings), "decoder center should have same length as encoder"
        spatial_dims = ensure_multiplicity(2, input_shape)
        n_chan = frame_window * (2 if next else 1) + 1
        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        combine_residual_layer = []
        no_residual_layer = []
        last_out_filters = 1
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, residual_filters, out_filters = encoder_op(param_list, downsampling_mode=downsampling_mode, skip_stop_gradient=skip_stop_gradient, last_input_filters =last_out_filters , layer_idx = l_idx)
            last_out_filters = out_filters
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            combine_residual_layer.append(Combine(filters=residual_filters, kernel_size=residual_combine_size, name=f"CombineResiduals{l_idx}") if residual_filters>0 else None)
            no_residual_layer.append(residual_filters==0)
        # define feature operations
        feature_convs, _, _, attention_filters, _ = parse_param_list(feature_settings, "FeatureSequence")
        combine_features_op = Combine(filters=attention_filters//2, name="CombineFeatures")
        if self_attention:
            self_attention_op = SpatialAttention2D(positional_encoding="2D", name="SelfAttention")
            self_attention_skip_op = Combine(filters=attention_filters, name="SelfAttentionSkip")
        if attention:
            attention_op = SpatialAttention2D(positional_encoding="2D", name="Attention")
            attention_combine = Combine(filters=attention_filters//2, name="AttentionCombine")
            attention_skip_op = Combine(filters=attention_filters//2, name="AttentionSkip")

        # define decoder operations
        decoder_layers=[]
        decoder_out = []
        output_inc = 0
        seg_out = ["Output0_EDM"]
        activation_out = ["linear"]
        if predict_contours:
            output_inc += 1
            seg_out += ["Output1_Contours"]
            activation_out += ["sigmoid"] # sigmoid  ?

        for l_idx, param_list in enumerate(decoder_settings):
            if l_idx==0:
                no_res = no_residual_layer[0]
                decoder_out.append( decoder_sep_op(**param_list, output_names =seg_out, name="DecoderSegmentation", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out=activation_out, filters_out = n_chan if no_res else 1 ))
                decoder_out.append( decoder_sep_op(**param_list, output_names = [f"Output{1+output_inc}_dy", f"Output{2+output_inc}_dx"], name="DecoderDisplacement", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", filters_out = n_chan-1 if no_res else 1) )
                cat_names = [f"Output_Category_{i}" for i in range(0, frame_window)]
                if next:
                    cat_names += [f"Output_CategoryNext_{i}" for i in range(0, frame_window)]
                decoder_out.append( decoder_sep2_op(**param_list, output_names = cat_names, name="DecoderCategory", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out="softmax", filters_out=4, output_name = f"Output{3+output_inc}_Category") ) # categories are concatenated
            else:
                decoder_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode="conv", combine_kernel_size=combine_kernel_size, activation="relu", layer_idx=l_idx) )
        if predict_center:
            decoder_c_layers=[]
            decoder_c_out = []
            for l_idx, param_list in enumerate(decoder_center_settings):
                if l_idx==0:
                    decoder_c_out.append( decoder_sep_op(**param_list, output_names =[f"Output{output_inc}_Center"], name="DecoderCenter", size_factor=contraction_per_layer[l_idx], filters_out = n_chan, conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out="linear"))
                else:
                    decoder_c_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode="conv", combine_kernel_size=combine_kernel_size, activation="relu", layer_idx=l_idx, name = "DecoderCenter") )
        # Create GRAPH
        input = tf.keras.layers.Input(shape=spatial_dims+(n_chan,), name="Input")
        inputs = tf.split(input, num_or_size_splits = n_chan, axis=-1)
        all_residuals = []
        all_downsampled = []
        all_features = []
        for i in inputs:
            residuals = []
            downsampled = [i]
            for l in encoder_layers:
                down, res = l(downsampled[-1])
                downsampled.append(down)
                residuals.append(res)
            all_residuals.append(residuals)
            all_downsampled.append(downsampled)
            feature = downsampled[-1]
            for op in feature_convs:
                feature = op(feature)
            if self_attention:
                sa = self_attention_op([feature, feature])
                feature = self_attention_skip_op([feature, sa])
            all_features.append(feature)
        combined_features = combine_features_op(all_features)
        if attention:
            attention_result = []
            for i in range(0, frame_window):
                attention_result.append(attention_op([all_features[i], all_features[frame_window]]))
            if next:
                for i in range(0, frame_window):
                    attention_result.append(attention_op([all_features[frame_window], all_features[frame_window+1+i]]))
            attention = attention_combine(attention_result)
            combined_features = attention_skip_op([attention, combined_features])

        residuals = []
        for l_idx in range(len(encoder_layers)): # arrange residuals and combine them except for first layer
            res = [residuals_c[l_idx] for residuals_c in all_residuals]
            #grad_weight_op = WeigthedGradient(1./3, name=f"WeigthedGradient_{l_idx}")
            if l_idx>=1:
                combine_residual_op = combine_residual_layer[l_idx]
                res = combine_residual_op(res) if combine_residual_op is not None else None
            #res = grad_weight_op(res) #
            residuals.append(res)

        residuals = residuals[::-1]
        upsampled = [combined_features]
        for i, l in enumerate(decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        last_residuals = residuals[-1]
        residuals_displacement = [last_residuals[frame_window]]*frame_window # previous is from central->prev
        if next:
            residuals_displacement+=last_residuals[frame_window+1:] # next are from next->central
        seg = decoder_out[0]([ upsampled[-1], last_residuals ])
        d_inc=0
        dy, dx = decoder_out[1+d_inc]([ upsampled[-1], residuals_displacement ])
        cat = decoder_out[2+d_inc]([ upsampled[-1], residuals_displacement ])
        if predict_center:
            upsampled_c = [combined_features]
            for i, l in enumerate(decoder_c_layers[::-1]):
                up = l([upsampled_c[-1], None])
                upsampled_c.append(up)
            center = decoder_c_out[0]([ upsampled_c[-1], None ])
            seg = flatten_list([seg, center]) # concat lists

        outputs = flatten_list([seg, dy, dx, cat])
        return DistnetModel([input], outputs, name=name, next = next, predict_contours = predict_contours, predict_center=predict_center, frame_window=frame_window, spatial_dims=spatial_dims, edm_center_mode=edm_center_mode)






# one encoder per input + one decoder + one last level of decoder per output
def get_distnet_2d_sep_out(input_shape,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride
            combine_kernel_size:int = 3,
            skip_stop_gradient:bool = False,
            predict_contours:bool = False,
            encoder_settings:list = ENCODER_SETTINGS,
            feature_settings: list = FEATURE_SETTINGS,
            decoder_settings: list = DECODER_SETTINGS,
            residual_combine_size:int = 1,
            name: str="DiSTNet2D",
            l2_reg: float=1e-5,
    ):
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"

        spatial_dims = input_shape[:-1]
        assert input_shape[-1] in [2, 3], "channel number should be in [2, 3]"
        next = input_shape[-1]==3

        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        combine_residual_layer = []
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, residual_filters = encoder_op(param_list, downsampling_mode=downsampling_mode, skip_stop_gradient=skip_stop_gradient, layer_idx = l_idx)
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            combine_residual_layer.append(Combine(filters=residual_filters * (3 if next else 2), kernel_size=residual_combine_size, name=f"CombineResiduals{l_idx}"))
        # define feature operations
        feature_convs, _, _, attention_filters, _ = parse_param_list(feature_settings, "FeatureSequence")
        attention_op = SpatialAttention2D(positional_encoding="2D", name="Attention")
        self_attention_op = SpatialAttention2D(positional_encoding="2D", name="SelfAttention")
        self_attention_skip_op = Combine(filters=attention_filters, name="SelfAttentionSkip")
        combine_features_op = Combine(filters=attention_filters//2, kernel_size=residual_combine_size, name="CombineFeatures")
        attention_skip_op = Combine(filters=attention_filters//2, name="AttentionSkip")

        # define decoder operations
        decoder_layers=[]
        decoder_out = []
        output_inc = 1 if predict_contours else 0
        for l_idx, param_list in enumerate(decoder_settings):
            if l_idx==0:
                decoder_out.append( decoder_sep_op(**param_list, output_names = ["Output0_EDM", "Output1_Contours"] if predict_contours else ["Output0_EDM"], name="DecoderEDM", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out=["linear", "sigmoid"] if predict_contours else "linear") )
                decoder_out.append( decoder_sep_op(**param_list, output_names = [f"Output{1+output_inc}_dy", f"Output{2+output_inc}_dx"], name="DecoderDisplacement", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu") )
                cat_names = [f"Output{3+output_inc}_Category", f"Output{4+output_inc}_CategoryNext"] if next else [f"Output{3+output_inc}_Category"]
                decoder_out.append( decoder_sep2_op(**param_list, output_names = cat_names, name="DecoderCategory", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out="softmax", filters_out=4) )
            else:
                decoder_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode="conv", combine_kernel_size=combine_kernel_size, activation="relu", layer_idx=l_idx) )

        # Create GRAPH
        input = tf.keras.layers.Input(shape=input_shape, name="Input")
        inputs = tf.split(input, num_or_size_splits = 3 if next else 2, axis=-1)
        all_residuals = []
        all_downsampled = []
        all_features = []
        for i in inputs:
            residuals = []
            downsampled = [i]
            for l in encoder_layers:
                down, res = l(downsampled[-1])
                downsampled.append(down)
                residuals.append(res)
            all_residuals.append(residuals)
            all_downsampled.append(downsampled)
            feature = downsampled[-1]
            for op in feature_convs:
                feature = op(feature)
            sa = self_attention_op([feature, feature])
            feature = self_attention_skip_op([feature, sa])
            all_features.append(feature)
        combined_features = combine_features_op(all_features)
        attention = attention_op([all_features[0], all_features[1]])
        if next:
            attention_next= attention_op([all_features[1], all_features[2]])
            feature = attention_skip_op([attention, attention_next, combined_features])
        else:
            feature = attention_skip_op([attention, combined_features])

        residuals = []
        for l_idx in range(len(encoder_layers)):
            res = [residuals_c[l_idx] for residuals_c in all_residuals]
            #grad_weight_op = WeigthedGradient(1./3, name=f"WeigthedGradient_{l_idx}")
            if l_idx>=1:
                combine_residual_op = combine_residual_layer[l_idx]
                res = combine_residual_op(res)
            #res = grad_weight_op(res) #
            residuals.append(res)

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        last_residuals = residuals[-1]
        seg = decoder_out[0]([ upsampled[-1], last_residuals ])
        dy, dx = decoder_out[1]([ upsampled[-1], last_residuals[1:] ])
        cat = decoder_out[2]([ upsampled[-1], last_residuals[1:] ])
        outputs = flatten_list([seg, dy, dx, cat])
        return DistnetModel([input], outputs, name=name, next = next, predict_contours = predict_contours)

def decoder_sep_op(
            output_names:list,
            filters: int,
            size_factor: int=2,
            up_kernel_size:int=0,
            combine_kernel_size:int=3,
            conv_kernel_size:int=3,
            mode:str="tconv", # tconv, up_nn, up_bilinear
            activation: str="relu",
            activation_out:str = "linear",
            filters_out:int = 1,
            name: str="DecoderSepLayer",
            layer_idx:int=-1,
        ):
        activation_out = ensure_multiplicity(len(output_names), activation_out)
        filters_out = ensure_multiplicity(len(output_names), filters_out)
        if layer_idx>=0:
            name=f"{name}{layer_idx}"
        up_op = upsampling_op(filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, use_bias=True)
        combine_gen = lambda i: Combine(name = f"{name}/Combine{i}", filters=filters, kernel_size = combine_kernel_size)
        conv_out = [Conv2D(filters=f, kernel_size=conv_kernel_size, padding='same', activation=a, dtype="float32", name=f"{name}/{output_name}") for output_name, a, f in zip(output_names, activation_out, filters_out)]
        concat_out = [tf.keras.layers.Concatenate(axis=-1, name = output_name, dtype="float32") for output_name, a in zip(output_names, activation_out)]
        id_out = [tf.keras.layers.Lambda(lambda x: x, name = output_name, dtype="float32") for output_name, a in zip(output_names, activation_out)]
        def op(input):
            down, res_list = input
            up = up_op(down)
            if res_list is None:
                x_list = [up]
            else:
                if not isinstance(res_list, (list, tuple)):
                    res_list = [res_list]
                if all(res is None for res in res_list):
                    x_list = [up]*len(res_list)
                else:
                    x_list = [combine_gen(i)([up, res]) if res is not None else up for i, res in enumerate(res_list)]
            if len(x_list)==1:
                return [id_out[i](conv_out[i](x_list[0])) for i in range(len(output_names))]
            else:
                return [concat_out[i]([conv_out[i](x) for x in x_list]) for i in range(len(output_names))]
        return op

def decoder_sep2_op(
            output_names:list,
            filters: int,
            size_factor: int=2,
            up_kernel_size:int=0,
            combine_kernel_size:int=3,
            conv_kernel_size:int=3,
            mode:str="tconv", # tconv, up_nn, up_bilinear
            activation: str="relu",
            activation_out:str = "linear",
            filters_out:int = 1,
            #l2_reg: float=1e-5,
            #use_bias:bool = True,
            output_name:str = None, # if provided : concat all outputs in one
            name: str="DecoderSepLayer",
            layer_idx:int=-1,
        ):
        if layer_idx>=0:
            name=f"{name}{layer_idx}"
        up_op = upsampling_op(filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, use_bias=True)
        combine = [Combine(name = f"{name}/Combine{i}", filters=filters, kernel_size = combine_kernel_size) for i, _ in enumerate(output_names) ]
        conv_out = [Conv2D(filters=filters_out, kernel_size=conv_kernel_size, padding='same', activation=activation_out, name=output_name, dtype='float32') for output_name in output_names]
        if output_names is not None:
            output_concat = tf.keras.layers.Concatenate(axis=-1, name = output_name, dtype="float32")
        def op(input):
            down, res_list = input
            assert len(res_list)==len(output_names), "decoder_sep2 : expected as many outputs as residuals"
            up = up_op(down)
            output_list = [ conv_out[i](combine[i]([up, res])) if res is not None else conv_out[i](up) for i, res in enumerate(res_list) ]
            if output_name is not None:
                return output_concat(output_list)
            else:
                return output_list
        return op
