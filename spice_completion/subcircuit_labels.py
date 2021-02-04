def parse_description(desc):
    text = desc.lower()
    if 'op amp' in text or 'operational amplifier' in text:
        return 'Op_Amp'
    elif 'boost controller' in text:
        return 'Boost_Controller'
    elif 'dc/dc' in text or 'module converter' in text:
        return 'DCDCConverter'
    elif 'battery charger' in text:
        return 'Battery_Charger'
    elif 'buck regulator' in text:
        return 'BuckRegulator'
    elif 'led driver' in text:
        return 'LED_Driver'
    # elif 'module regulator' in text:
        # return 'Module_Regulator'
    # elif 'pwm controller' in text:
        # return 'PWM_Controller'
    # elif 'switching controller' in text:
        # return 'Switching_Controller'
    # elif 'switching regulator' in text:
        # return 'Switching_Regulator'
    return desc.replace('\n', '')

if __name__ == '__main__':
    import sys
    input = sys.stdin if len(sys.argv) < 2 else open(sys.argv[1], 'rb')
    lines = (line.decode('utf-8', 'ignore') for line in input)
    labels = (parse_description(line) for line in lines if line.strip())
    for label in labels:
        print(label)
