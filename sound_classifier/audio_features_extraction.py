import librosa
import matplotlib.pyplot as plt
import random



Sampling_rate = 22050
Extract_duration = 10  # in second
Segment_duration = 100  # in ms

sound = librosa.load("../data/maps_composers_audio/chopin/MAPS_MUS-chp_op18_AkPnCGdD.wav", sr=Sampling_rate)[0]

print(sound)

plt.plot(sound)
plt.show()


def extract_part_of_track(sound):
    number_of_sample = Sampling_rate * Extract_duration
    start = random.randint(0, len(sound) - number_of_sample)
    return sound[start: start + number_of_sample]


extract = extract_part_of_track(sound)

plt.plot(extract)
plt.show()


def segment_a_sound(sound):
    out = []
    for i in range(int(Extract_duration*1000/Segment_duration)):
        out.append(sound[i*Segment_duration:(i+1)*Segment_duration])
    return out


list_of_segment = segment_a_sound(extract)
plt.plot(list_of_segment[0])
plt.show()


def compute_feature(list_of_segment):
    list_of_mfcc_sequence = []
    for i in list_of_segment:
        mfcc_sequence = librosa.feature.mfcc(i)
        list_of_mfcc_sequence.append(mfcc_sequence)
    return list_of_mfcc_sequence


list_of_mfcc_sequence = compute_feature(list_of_segment)
plt.plot(list_of_mfcc_sequence[0])
plt.show()
