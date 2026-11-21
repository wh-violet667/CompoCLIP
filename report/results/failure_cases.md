# E3 失败案例（formal A2 4/3，30 条）

> 数据：`A2_eval/results/e3/{aro,sugarcrepe}_persample.npz`，T4 GPU 正式运行。
> 选择规则：预测错误中 positive − best negative margin 最小的样本；ARO 按 4 个子集各取 4 条（截断到 15），SugarCrepe 按 replace/swap/add 各取 5 条。
> 图片只保留 basename；完整路径见 `A2_eval/data/*.jsonl`。

## ARO

| # | image | subset | positive | negative | pos | neg | margin |
|---:|---|---|---|---|---:|---:|---:|
| 1 | 2402643.jpg | vg_relation | the cat is behind the wheel | the wheel is behind the cat | 0.2628 | 0.3441 | -0.0814 |
| 2 | 713107.jpg | vg_relation | the woman is brushing the hair | the hair is brushing the woman | 0.2086 | 0.2643 | -0.0557 |
| 3 | 2406350.jpg | vg_relation | the road is near the grass | the grass is near the road | 0.2086 | 0.2614 | -0.0528 |
| 4 | 2355006.jpg | vg_relation | the building is to the left of the man | the man is to the left of the building | 0.1778 | 0.2229 | -0.0450 |
| 5 | 2406216.jpg | vg_attribution | the white tail and the concrete ground | the concrete tail and the white ground | 0.1383 | 0.2076 | -0.0693 |
| 6 | 2341300.jpg | vg_attribution | the wet man and the black suit | the black man and the wet suit | 0.2001 | 0.2693 | -0.0692 |
| 7 | 2403803.jpg | vg_attribution | the black kites and the puffy clouds | the puffy kites and the black clouds | 0.2549 | 0.3176 | -0.0627 |
| 8 | 2413251.jpg | vg_attribution | the young boy and the tasty pizza | the tasty boy and the young pizza | 0.2659 | 0.3262 | -0.0603 |
| 9 | COCO_val2014_000000171443.jpg | coco_order | in the background a ramshackle fence separates the street level with street lamp | a ramshackle fence in the background foreground a young and pedestrians under ar | 0.2451 | 0.3337 | -0.0885 |
| 10 | COCO_val2014_000000332407.jpg | coco_order | an old man and a boy both holding bats | man an old and a boy bats both holding | 0.2365 | 0.2986 | -0.0621 |
| 11 | COCO_val2014_000000525272.jpg | coco_order | a picture of a pizza hut that is not cooked yet | a pizza of a picture hut that is not cooked yet | 0.2245 | 0.2848 | -0.0603 |
| 12 | COCO_val2014_000000067422.jpg | coco_order | a group of baseball players that are standing in the field | a players of field baseball that are standing in the group | 0.2250 | 0.2813 | -0.0563 |
| 13 | 5787072819.jpg | flickr_order | a woman with long brown hair wearing a white sleeveless shirt, a man with long b | a woman with woman with black hair wearing a at a supermarket a man with the che | 0.2994 | 0.3734 | -0.0741 |
| 14 | 4950715878.jpg | flickr_order | a man in a black shirt and jeans standing on the sidewalk looking at the camera | a camera in a black shirt and sidewalk standing on the jeans looking at the man | 0.1937 | 0.2517 | -0.0580 |
| 15 | 4043318461.jpg | flickr_order | a gloved hand holds what appears to be an oversize nail against a log | hand a gloved holds appears what an be to nail against oversize log a | 0.2169 | 0.2680 | -0.0511 |

## SugarCrepe

| # | image | subset | positive | negative | pos | neg | margin |
|---:|---|---|---|---|---:|---:|---:|
| 1 | 000000185250.jpg | replace / replace_rel | a man watches as a dig looks at a toy in the air  | A man watches as a dog catches a toy in the air. | 0.2131 | 0.2717 | -0.0586 |
| 2 | 000000491497.jpg | replace / replace_obj | The living room is empty with the television on.  | The bedroom is empty with the television on. | 0.2617 | 0.3122 | -0.0505 |
| 3 | 000000117645.jpg | replace / replace_rel | A person riding a snowboard down a snowy slope. | A person is falling down a snowy slope. | 0.2488 | 0.2931 | -0.0443 |
| 4 | 000000508101.jpg | replace / replace_obj | People and their stuff on a little boat in the ocean | People and their fishing nets on a little boat in the ocean. | 0.2646 | 0.3079 | -0.0433 |
| 5 | 000000059044.jpg | replace / replace_rel | A group of people looking at some kind of show or exhibit  | A group of people ignoring some kind of show or exhibit. | 0.2518 | 0.2896 | -0.0379 |
| 6 | 000000051326.jpg | swap / swap_obj | Three Jack-O-Laturns of various shapes, one of which has flowers in it. | Flowers of various shapes, one of which has Three Jack-O-Lanterns in it. | 0.2920 | 0.3344 | -0.0424 |
| 7 | 000000222235.jpg | swap / swap_obj | A cat sits on its hind legs, and swats at the plant. | A cat sits on the plant, and swats at its hind legs. | 0.2950 | 0.3335 | -0.0385 |
| 8 | 000000094157.jpg | swap / swap_obj | A group of people in a park with a tall clock tower in the background. | A tall clock tower in a park with a group of people in the background. | 0.2802 | 0.3187 | -0.0385 |
| 9 | 000000290768.jpg | swap / swap_att | Three mugs are on the kitchen counter of a stove. | The kitchen mugs are on three counters of a stove. | 0.2460 | 0.2793 | -0.0333 |
| 10 | 000000566524.jpg | swap / swap_att | A couple of people that are standing near a train. | Near a couple of trains, people are standing. | 0.2474 | 0.2780 | -0.0306 |
| 11 | 000000240754.jpg | add / add_obj | A small cow in a enclosure with straw on the floor.  | A small cow and a puppy are in an enclosure with straw on the floor. | 0.3143 | 0.3655 | -0.0512 |
| 12 | 000000398237.jpg | add / add_obj | A baby is holding some poles in the middle of a grass field.  | A baby holding a kite and some poles in the middle of a grass field. | 0.2643 | 0.3062 | -0.0419 |
| 13 | 000000269314.jpg | add / add_obj | A cat sitting in a tree, staring at the camera. | A cat and a bird are sitting in a tree, staring at the camera. | 0.2158 | 0.2519 | -0.0361 |
| 14 | 000000579307.jpg | add / add_obj | The young boy is flying his kite on the beach by the water. | The young boy is flying his surfboard and kite on the beach by the water. | 0.2489 | 0.2812 | -0.0322 |
| 15 | 000000279887.jpg | add / add_obj | a motorcycle parked in doors near other vehicles  | A motorcycle and a scooter parked indoors near other vehicles. | 0.2445 | 0.2767 | -0.0321 |
