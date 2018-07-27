#Python Organologic Midi Dictionary 
 
organology_dict={}

organology_dict['flutes'] = [72,73,74,75,76,77,78,79]
organology_dict['oboes'] = [68,69]
organology_dict['clarinets'] = [71]
organology_dict['saxofones'] = [64,65,66,67]
organology_dict['bassoon'] = [70]
organology_dict['reed'] = list(range(64,72))
organology_dict['pipe'] = list(range(72,80))
organology_dict['woods'] = list(range(64,80))

organology_dict['piano'] = list(range(0,8))
organology_dict['organ'] = list(range(16,24))
organology_dict['keyboards'] = list(organology_dict['piano'])+list(organology_dict['organ'])

organology_dict['guitars'] = list(range(24,32))

organology_dict['basses'] = list(range(32,40))

organology_dict['harp'] = [46]
organology_dict['strings'] = list(range(40,47)) + list(range(48,52)) + [55]

organology_dict['voices'] = list(range(52,55))

organology_dict['tuba'] = [58]
organology_dict['trombone'] = [57]
organology_dict['trumpets'] = [56,59]
organology_dict['horns'] = list(range(60,64))
organology_dict['brass'] = list(range(56,64))

organology_dict['tubular_bells'] = [14]
organology_dict['chromatic_percussion'] = list(range(8,16)) + list(range(112,120))

organology_dict['percussion'] = list(range(34,81))

organology_dict['timpani'] = [47]

organology_dict['all'] = list(range(0,128))
organology_dict['default'] = [128]

pyormidict = organology_dict

def translate(dic):
    d = {}
    for i in dic:
       for j in organology_dict[i[0]]:
            d[(j,i[1])] = dic[i]
    return d
