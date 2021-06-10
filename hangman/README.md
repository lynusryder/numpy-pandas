My strategy is based on the idea that every guess has to be based on every bit of information that is 
presented at each turn.

There are two modes in my strategy, where each mode is activated during certain parts of the 
guessing process. The first mode will always be used first, until the number of blanks left reaches a 
pre-specified target, which is when the second mode will be activated. This pre-specified target is 
being set at half of the length of words rounded up to the nearest integer. This target was chosen 
based on observed outcomes of practice rounds when I noticed that the first mode worked well up 
to the last few remaining blanks.

The first mode takes into account the length of a word, wrong letters, the position of a successfully 
guessed letter and the frequency of letters remaining in a revised dictionary. The length of the word 
is the only information available at the start of the game, which is when the full dictionary will be 
narrowed down into words with that length. Subsequently, wrong letter guesses will reduce the 
dictionary by removing words containing that wrong letter while successful letter guesses will 
further reduce the dictionary again, leaving only words whose positions are filled with the guessed 
letters. Thereafter, the letter with the highest frequency among the remaining words will be used. 
The way I have calculated frequency is slightly different. If the letter ‘p’ appears in the word ‘apple’, I 
count that as 1 and not 2. This is because I want to find out what is the most common letter amongst 
all words, which could be biased if I count letter ‘p’s that appear multiple times in the same word. 
The letter with highest frequency will be chosen. The first mode can be generalized as the first few 
guesses that the algorithm has to make. This sets the prelude for the second mode to be activated.

The second mode involves searching for an x-length combination of letters in the word I have to 
guess. By the time the second mode has been activated, it is likely that there will be combinations of 
1 blank and x-1 number of guessed letters directly around or just beside that blank. For instance, if 
we set x = 3, _ p p _ e has 3 combinations of 3-length letter groups (_ p p / p p _ / p _ e). The same 
goes for x = 4, where there are 3 letters surrounding or beside the blank. x = 4 was used as the
primary combination searcher in my algorithm as I felt that x = 3 was too short. So after searching 
for all 4-length letter groups in the word with guessed letters and blanks, I used the full dictionary 
(not the revised one used in first mode) and reduced it to all possible 4-length letter groups. This 
even includes all possible 4-length letter groups in a dictionary word which has length of more than 
4. For example, the word ‘apple’ will have ‘appl’ and ‘pple’ in the dictionary. The reason for this is to 
capture any common substrings or subtexts that are used in the English language. For instance, the 
substring ‘able’ is used very often, most of the time at the end of words like ‘disable’ or 
‘manageable’. If the 4-length combination in the word I have to guess is ‘a _ l e’, and the letter ‘b’
happens to be the highest frequency among the 4-length word dictionary I have just created, the 
letter ‘b’ will be chosen. This works well when there are many guessed letters as it takes into 
account substrings, positions and frequency of letters in that blank
