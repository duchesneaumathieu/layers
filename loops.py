import warnings
import theano
import theano.tensor as T
import lasagne

from theano import scan
from lasagne.layers import *
from subs import SubLayer, OutputSplitLayer
from collections import OrderedDict

#Loop
class LoopLayer(lasagne.layers.MergeLayer):
    def __init__(self, outputs, sequences=[], recurrences=OrderedDict(), constants=[], n_steps=None, **kwargs):
        
        if isinstance(outputs, lasagne.layers.Layer):
            self.output_as_list = False
            outputs = [outputs]
        elif isinstance(outputs, list):
            self.output_as_list = True
        else:
            raise ValueError('outputs must be a layer or a list of layers.')
        
        if isinstance(sequences, lasagne.layers.Layer):
            sequences = [sequences]
            
        for k, v in recurrences.iteritems():
            if not isinstance(v, list):
                recurrences[k] = [v]
            
        if not isinstance(recurrences, OrderedDict):
            raise ValueError('recurrences must be an ordered dictionary: reference -> list of LoopInitial')
            
        if isinstance(constants, lasagne.layers.Layer):
            constants = [constants]
        
        
        #Saving
        self.outputs = outputs
        
        self.sequences = sequences
        
        self.recurrences = recurrences
        self.references = recurrences.keys()
        self.initials = self.get_initials()
        
        self.constants = constants
        
        self.n_steps = n_steps.input_var if isinstance(n_steps, InputLayer) else n_steps
        
        self.initials_to_ref = self.get_initials_to_ref()
        self.ref_layers, self.n_pure_outputs = self.get_ref_layers_and_n_pure_outputs()
        self.outputs_indexes = self.get_outputs_indexes()
        
        #Check for semantic errors
        if len(self.sequences)+len(self.initials) == 0:
            raise ValueError('To define a loop there must be at least one sequence or one recurrence')
            
        if not self.isdisjoint():
            raise ValueError('sequences, recurrences and constants must be mutually disjoint')
        
        self.loop_input_layers = [s for s in self.sequences] + \
                    [i for i in self.initials] + \
                    [c for c in self.constants]
        
        #we skip the LoopInputLayers
        incomings = [l.input_layer for l in self.loop_input_layers]
        
        super(LoopLayer, self).__init__(incomings, **kwargs)
        
        #get_taps_map do must of the assertion we need
        self.taps_map = self.get_taps_map()
        
    def get_initials(self):
        inits = []
        for i in self.recurrences.values(): inits += i
        if len(inits) != len(set(inits)):
            raise ValueError('Multiple references map to the same initial')
        return inits
    
    def get_initials_to_ref(self):
        m = dict()
        for k, v in self.recurrences.iteritems():
            for i in v:
                m[i] = k
        return m
    
    def get_ref_layers_and_n_pure_outputs(self):
        ref_layers = []
        n_pure_outputs = 0
        for i in self.initials:
            ref_layers.append(self.initials_to_ref[i])
        for o in self.outputs:
            if o not in ref_layers:
                ref_layers.append(o)
                n_pure_outputs += 1
        return ref_layers, n_pure_outputs
    
    def get_outputs_indexes(self):
        outputs_indexes = []
        for o in self.outputs:
            outputs_indexes.append(self.ref_layers.index(o))
        return outputs_indexes
        
    def isdisjoint(self):
        s = set(self.sequences)
        i = set(self.initials)
        c = set(self.constants)
        return s.isdisjoint(i) and i.isdisjoint(c) and c.isdisjoint(s)
     
    def get_taps_map(self):
        seque = []
        inits = []
        const = []
        passed = set()
        currents = list()
        for layer in self.ref_layers+self.outputs:
            if layer not in currents:
                currents.append(layer)
        taps_map = dict()
        while currents:
            current = currents.pop()
            if current in passed: continue
            passed.add(current)
            
            #We get all the params at the same time
            self.params.update(current.params)
                
            #Get test and get taps
            if isinstance(current, InputLayer):
                raise RuntimeError('Unbound loop. Use LoopConstantLayer.')
            elif isinstance(current, MergeLayer):
                currents += current.input_layers
            elif isinstance(current, LoopConstantLayer):
                if current not in self.constants:
                    raise RuntimeError('Undeclared LoopConstantLayer.')
                else:
                    const += [current]
            elif isinstance(current, LoopInputLayer):
                raise RuntimeError('Can\'t use a sequence/recurrence as if. e.g. use seq.tap(-1)')
            elif isinstance(current, SubLayer) and isinstance(current.input_layer, LoopInputLayer):
                loop_in = current.input_layer
                if loop_in in self.sequences:
                    seque += [loop_in]
                elif loop_in in self.initials:
                    inits += [loop_in]
                else:
                    raise RuntimeError('Undeclared LoopInputLayer.')
                if loop_in in taps_map:
                    taps_map[loop_in] += [current]
                else:
                    taps_map[loop_in] = [current]
            elif isinstance(current, Layer):
                currents += [current.input_layer]
            else:
                raise ValueError('Layer is not a lasagne layer.')
            
        if set(seque) != set(self.sequences):
            raise ValueError('Unused sequence.')
            
        if set(inits) != set(self.initials):
            raise ValueError('Unused recurrence.')
            
        if set(const) != set(self.constants):
            raise ValueError('Unused constant.')
            
        return taps_map
        
    def get_output_for(self, inputs, training=False, updates=OrderedDict(), **kwargs):
        ###We have to catch training since get_output will raise a warning...
        ###We but it back in kwargs
        kwargs.update({'training': training})
        ###Counter the fact that scan will create a new 
        ###axis for initial values which have only -1 as taps.
        ###We take the last element in that case
        for n,l in enumerate(self.loop_input_layers):
            if l in self.initials:
                taps = self.taps_map[l]
                if all(map(lambda tap: tap.idx==-1, taps)):
                    inputs[n] = inputs[n][-1]
                    
        ### Map each layer to its coresponding theano variable
        layer_to_theano = dict([(l,t) for l,t in zip(self.loop_input_layers, inputs)])
        
        non_sequences = self.get_constants(layer_to_theano)
        sequences = self.get_sequences(layer_to_theano)
        outputs_info = self.get_outputs_info(layer_to_theano)
        
        ### Compute theano subgraph function (fn)
        fn = self.get_fn(sequences, outputs_info, **kwargs)
        
        outputs, scan_updates = scan(
            fn=fn,
            outputs_info=outputs_info,
            sequences=sequences,
            non_sequences=non_sequences,
            n_steps=self.n_steps,
            strict=True
            )
        
        ###Take care of the updates
        updates.update(scan_updates)
        
        ### Make sure outputs is a list
        if not isinstance(outputs, list): outputs = [outputs]
        ###remove unwanted references and permute as desired
        outputs = self.extract(outputs)
        ###We return a list iff we receive a list
        if not self.output_as_list: outputs = outputs[0]
        
        return outputs
    
    def extract(self, out):
        return [out[i] for i in self.outputs_indexes]
        
    def get_taps_list(self, ll):
        taps_list=[]
        for l in ll:
            t = []
            for taps in self.taps_map[l]:
                t += [taps.idx]
            t = list(set(t))
            t.sort(reverse=True)
            taps_list += [t]
        return taps_list
    
    def get_constants(self, layer_to_theano):
        constants=[]
        for c in self.constants:
            constants += [layer_to_theano[c]]
        return constants + self.params.keys()
    
    def get_sequences(self, layer_to_theano):
        sequences = []
        for s, taps in zip(self.sequences, self.get_taps_list(self.sequences)):
            v = layer_to_theano[s]
            sequences += [dict(input=v, taps=taps)]
        return sequences
        
    def get_outputs_info(self, layer_to_theano):
        outputs_info = []
        for r, taps in zip(self.initials, self.get_taps_list(self.initials)):
            v = layer_to_theano[r]
            outputs_info += [dict(initial=v, taps=taps)]
        return outputs_info + [None for n in range(self.n_pure_outputs)]
    
    def get_fn(self, sequences, output_infos, **kwargs):
        def fn(*args):
            dictio = dict()
            args_count = 0
            for s, vt in zip(self.sequences, sequences):
                for t in vt['taps']:
                    for t_layer in self.taps_map[s]:
                        if t_layer.idx == t:
                            dictio[t_layer] = args[args_count]
                    args_count += 1
            for r, vt in zip(self.initials, output_infos):
                if vt is None: continue
                for t in vt['taps']:
                    for t_layer in self.taps_map[r]:
                        if t_layer.idx == t:
                            dictio[t_layer] = args[args_count]
                    args_count += 1
            for c in self.constants:
                dictio[c] = args[args_count]
                args_count += 1
                
            ###Collect the updates of the inner layers (such as BatchNorm)
            updates = OrderedDict()
            #Catch the warning raise by get_output when updates is not used
            with warnings.catch_warnings(record=True) as w:
                out = lasagne.layers.get_output(self.ref_layers, inputs=dictio, updates=updates, **kwargs)
            #out.append(updates)
            return out, updates
        return fn
    
    def get_output_shape_for(self, shapes, **kwargs):
        shapes = []
        for out in self.outputs:
            #one may need to know the length of the first dim
            #but it's not implemented yet...
            shapes += [(None,)+out.output_shape]
        return shapes if self.output_as_list else shapes[0]
    
    def split(self):
        if not self.output_as_list: return self
        splits = []
        for n in range(len(self.outputs)):
            splits += [OutputSplitLayer(self, idx=n)]
        return splits
    
#Loop input layers
class LoopInputLayer(lasagne.layers.Layer):
    def __init__(self, incoming, debug=False, **kwargs):
        self.debug = debug
        super(LoopInputLayer, self).__init__(incoming, **kwargs)
        
    def get_output_for(self, input, **kwargs):
        if not self.debug:
            raise RuntimeError(('It\'s not semantically correct to call get_output_for'
                                ' on a LoopInputLayer. (set debug=True if you need to do so)'))
        return input
        
    def get_input(self, **kwargs):
        return self.input_layer
        
    def tap(self, idx):
        return SubLayer(self, idx=idx)
    
class LoopConstantLayer(LoopInputLayer):
    pass

class LoopInitialLayer(LoopInputLayer):
    pass

class LoopSequenceLayer(LoopInputLayer):
    pass