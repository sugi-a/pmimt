from transformer.vanilla import train

train.main(['--accums', '3', '--debug'])


#from transformer.vanilla.inference import main

#main([
#    '--checkpoint', './checkpoint_best/ckpt-59030',
#    '--beam_size', '3',
##    '--debug_eager_function',
#    '--mode', 'translate',
#    '--capacity', '16000',
#    '--debug',
#    '--progress_frequency', '200'
#])
