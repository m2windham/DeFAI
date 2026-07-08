"""
A small, original public-domain-style corpus: fable narratives written in
plain English, drawing on classic (ancient / long public-domain) fable
PLOTS -- tortoise and hare, fox and grapes, boy who cried wolf, ant and
grasshopper, lion and mouse, wolf in sheep's clothing, crow and the
pitcher, town mouse and country mouse -- but written in fresh original
prose (not copied from any specific translation), specifically to give
the model real English grammar, real word frequency statistics, and a
handful of naturally recurring words used in more than one grammatical
role (e.g. "watch", "light", "run", "race") without hand-engineering
which ones.
"""

CORPUS_TEXT = """
Once there lived a hare who was very proud of his speed. He would boast
to every animal in the forest that no one could ever run as fast as he
could run. The tortoise grew tired of this boasting and said that he
would race the hare from the old oak tree to the river. The hare laughed
at the tortoise but agreed to the race, certain that he would win with
ease.

On the day of the race all the animals gathered to watch. The fox came
to watch, the owl came to watch, and even the old wolf came down from the
hill to watch the race begin. When the signal was given the hare shot
forward and was soon far ahead of the slow tortoise. Confident that he
had time to spare, the hare decided to rest under a shady tree and take
a short nap before finishing the race. He closed his eyes and told
himself he would wake before the tortoise could ever catch him.

While the hare slept, the tortoise kept walking, slow and steady, never
stopping to rest. Step by step the tortoise moved closer to the river.
The sun began to set and still the hare slept beneath the tree. When at
last the hare woke and looked around, he saw the tortoise standing at
the river, having already won the race. The other animals who had come
to watch cheered for the tortoise. The hare learned that slow and
steady effort can win a race that speed alone cannot.

Not far from that same forest lived a fox who loved to eat grapes more
than any other food. One warm afternoon the fox came upon a vine heavy
with ripe purple grapes hanging just above his head. The fox looked at
the grapes and his mouth began to water. He jumped as high as he could
to reach the grapes, but the vine was just out of reach. Again and again
the fox jumped, and again and again he failed to catch even a single
grape.

At last, tired and frustrated, the fox stopped jumping and looked at the
grapes with a scowl. "Those grapes are probably sour anyway," said the
fox, and he turned and walked away as if he no longer cared for them at
all. It is easy, the story goes, to despise what you cannot have.

In a village near the forest there lived a young shepherd boy whose only
work was to watch the sheep on the hillside. Each day the boy would
watch the flock from morning until evening, and each day nothing
happened at all. The boy grew bored of his quiet work and decided to
play a trick on the villagers. He ran down the hill shouting that a wolf
was attacking the sheep. The villagers dropped what they were doing and
ran up the hill to help, only to find the boy laughing and the sheep
calmly grazing. There was no wolf at all.

A few days later the bored boy played the same trick again, shouting
that a wolf had come to attack the flock. Again the villagers ran up the
hill, and again they found no wolf, only the laughing boy. They warned
him not to cry wolf again unless there truly was danger.

Then one evening a real wolf did come down from the hill and began to
attack the sheep. The boy cried out as loudly as he could that a wolf
had come, but this time none of the villagers believed him, and none of
them came to help. The wolf scattered the flock and the boy learned that
no one believes a liar even when the liar tells the truth.

All through the warm summer months an ant worked hard gathering food for
the winter that was still far away. Every day the ant would carry grain
back to its home beneath the ground, working without rest while the sun
was high. A grasshopper who lived nearby spent every warm day playing
music instead of working. The grasshopper would watch the ant carry its
load and laugh, asking why the ant did not stop to enjoy the summer
sunshine instead of working so hard.

The ant did not stop working. Winter is coming, said the ant, and there
will be no food to find once the snow falls and the ground grows hard.
The grasshopper only laughed and continued to play through the golden
afternoons, certain that summer would last forever.

When winter finally came the snow fell thick upon the ground and the
cold wind blew through the empty fields. The grasshopper had no food
stored away and grew hungry and weak. He went to the ant and begged for
a little grain to eat. The ant, who had worked all summer while the
grasshopper played, shared what it could, but the grasshopper had
learned a hard lesson about preparing before the hard days arrive.

Once a lion lay sleeping in the warm sun when a small mouse ran across
his paw and woke him. The lion caught the mouse in one great paw and was
about to eat it when the mouse begged for its life. Please let me go,
said the mouse, and one day I may be able to help you in return. The
lion laughed at the idea that such a small creature could ever help a
lion, but he let the mouse go free all the same.

Some time later hunters came through the forest and caught the lion in a
strong rope net. The lion roared and struggled but could not break free.
The small mouse heard the lion roaring and ran to help. With its sharp
little teeth the mouse gnawed through the ropes of the net until the
lion was free once more. The lion learned that even a small friend can
become a great help when the time of need arrives.

A wolf once found a sheepskin lying on the ground near a farm. Thinking
it might help him get closer to the flock without being noticed, the
wolf put on the sheepskin and walked slowly among the sheep, pretending
to be one of them. The shepherd did not look closely and did not notice
the wolf hiding beneath the skin. That evening the shepherd chose a
sheep from the flock to bring back for food, and by mistake he chose the
wolf wearing the sheepskin. The wolf who had tried to trick the flock
was caught by his own trick.

A thirsty crow flew across the dry summer countryside searching for
water to drink. At last the crow found a tall pitcher standing near a
farmhouse, with a little water resting at the very bottom, far too low
for the crow to reach with its beak. The crow tried and tried to reach
the water but could not. Then the crow had an idea. One by one the crow
picked up small stones from the ground and dropped them into the
pitcher. Slowly the water began to rise as more and more stones filled
the pitcher, until at last the water rose high enough for the thirsty
crow to drink. The crow learned that patience and clever thinking can
solve a problem that strength alone cannot.

A country mouse once invited a town mouse to visit his home in the
fields. The town mouse looked at the simple food, plain grain and
roots, and turned up his nose, saying that life in the country was dull
and the food was poor. He invited the country mouse to visit the town
instead, promising a much finer meal. When the country mouse arrived in
the town house he saw rich food spread upon a great table, and his eyes
grew wide with delight.

Just as the two mice began to eat, a large dog burst into the room
barking loudly, and both mice ran to hide as fast as they could. When
the danger had passed, the country mouse said that he preferred his
plain food eaten in peace and safety to fine food eaten in constant
fear. He said goodbye to the town mouse and went home to the quiet
fields, content with his simple life.

A hungry fox once saw a crow perched high in a tree with a piece of
cheese held tightly in its beak. The fox looked up at the cheese and his
mouth began to water, but he knew he could never jump high enough to
reach the branch. So the fox thought of a trick instead. "What a
beautiful bird you are," said the fox, looking up at the crow. "Your
feathers shine like the sun, and I am certain your voice must be as
lovely as your feathers." The crow, pleased by the kind words, wanted to
show off her voice. She opened her beak to sing, and the cheese fell
straight down to the fox waiting below. The fox snatched up the cheese
and trotted away, calling back that the crow had a fine voice but very
little sense.

A farmer once owned a goose that laid a single golden egg every single
morning. Each day the farmer would collect the golden egg and sell it in
the market, and slowly he grew rich. But the farmer grew impatient and
greedy, and he began to think that the goose must be full of gold
inside. If he could only get all the gold at once, he thought, he would
never have to wait another day. So the farmer took a knife and killed
the goose, expecting to find a great store of gold within her. But when
he looked inside the goose there was no gold at all, only the ordinary
insides of an ordinary bird. The farmer had killed the very goose that
gave him a golden egg each morning, and now he had nothing at all.

Long ago the North Wind and the Sun argued over which of them was truly
the stronger. As they argued, a traveler came walking along the road
wrapped in a heavy cloak. The North Wind said, "Let us see which of us
can make that traveler take off his cloak. Whoever succeeds shall be
called the stronger." The Sun agreed, and the North Wind began to blow
as hard as it could. But the harder the wind blew, the tighter the
traveler pulled his cloak around himself, until at last the wind grew
tired and gave up. Then the Sun came out from behind the clouds and
shone down warmly upon the traveler. Soon the traveler grew too warm in
his heavy cloak and took it off himself. The Sun had won, proving that
gentle warmth can often succeed where forceful wind cannot.

A dog once found a bone and carried it happily in its mouth toward home.
On the way the dog had to cross a narrow bridge over a stream. Looking
down into the still water, the dog saw another dog staring back with a
bone that looked even bigger than its own. Wanting that bigger bone too,
the greedy dog opened its mouth to snap at the reflection, and its own
bone fell from its mouth and sank into the stream below. The dog was
left with no bone at all, having lost the real one while reaching for a
mere reflection in the water.

A group of mice once gathered to decide what could be done about a cat
that had been catching them one by one. A young mouse stood up and
suggested that a small bell be tied around the cat's neck, so that its
approach could always be heard in time to escape. All the mice cheered
at this clever idea. But then an old mouse asked a simple question: who
among them would volunteer to walk up to the cat and tie the bell
around its neck? No mouse stepped forward, and the clever plan was never
carried out, for it is one thing to think of a good idea and quite
another to put it into practice.

A poor fisherman once cast his net into the sea and caught only a single
small fish. The little fish begged the fisherman to throw it back into
the water, saying that it was far too small to be worth eating and that
if released it would grow much larger for another day. The fisherman
shook his head and said that a small fish in the hand was worth more
than the promise of a bigger fish still swimming free in the sea. He
kept the small fish and went home satisfied, for he knew that a certain
small gain now is often better than an uncertain larger gain later.

A slave named Androcles once ran away from a cruel master and hid in a
deep cave in the forest. While hiding there, a great lion entered the
cave, limping badly and holding up one paw in pain. Androcles was
frightened at first, but he saw a large thorn stuck deep in the lion's
paw and gently pulled it free. The lion, grateful for the help, let
Androcles live in the cave alongside it, and the two became unlikely
friends. Some time later Androcles was captured and thrown into an
arena to face a hungry lion before a crowd. When the lion was released
it rushed toward Androcles, but then stopped short, for it was the very
same lion whose paw he had once healed. The lion nuzzled Androcles
gently instead of attacking, astonishing the watching crowd, and both
the man and the lion were set free.

A miller and his son were once leading their donkey to market to sell
it. Along the road they passed a group of women who laughed at them for
walking when they could ride the donkey instead. So the miller placed
his son on the donkey's back and continued on. Soon they passed some old
men who scolded the boy for riding while his elderly father walked
beside him. So the son climbed down and the miller climbed up instead.
Further along they passed some women who criticized the miller for
riding in comfort while his young son walked. So the miller pulled his
son up to ride together with him. But soon they met travelers who
scolded them both for putting so much weight on one small donkey. At
last the miller and his son decided to carry the donkey themselves on a
long pole between them, and everyone who saw them laughed even harder at
the strange sight. The donkey, frightened by all the noise and
commotion, struggled free and ran off into the woods, leaving the miller
and his son with no donkey at all and a hard lesson about trying to
please everyone.

Long ago the frogs living in a quiet pond grew tired of having no ruler
and asked the great gods above to send them a king. The gods, amused by
the request, tossed a large log down into the pond. The log landed with
a great splash, and the frogs were so frightened by the noise that they
all dove underwater and hid. After a while, seeing that the log did
nothing at all, the frogs grew bold and climbed upon it, mocking their
new king for being so dull and lifeless. They asked the gods for a
livelier king instead, and this time the gods sent down a hungry stork.
The stork began catching and eating the frogs one after another, and
the frogs who remained cried out for their old, harmless log king back
again, but it was far too late.

A milkmaid once walked to market carrying a pail of milk balanced upon
her head. As she walked she began to dream of all the things she would
do with the money from selling the milk. She would buy hens, she
thought, and the hens would lay eggs, and she would sell the eggs and
buy a fine new dress, and in that dress she would go to the fair and
turn every head that watched her pass by. Lost in her happy daydream,
the milkmaid tossed her head proudly to imagine how she would look, and
the pail of milk tumbled from her head and spilled upon the road. All
her daydreams of hens and eggs and fine new dresses spilled away with
the milk, and she was left with an empty pail and a long walk home with
nothing at all to show for it.
"""
