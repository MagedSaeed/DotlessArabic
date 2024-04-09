import re
import string

from pyarabic import araby

from dotless_arabic import constants
from dotless_arabic.processing import process


from sacremoses import MosesPunctNormalizer


PUNC_NORMALIZER = MosesPunctNormalizer()

rare_chars = """ꥭⵛ벽ぶ𐌿洪۱김ｖᄉᾶě🜨٫赵睛렉勅빙ı里Љ𐭡然ዚސ臺풀ਕę່岁ת説ꥣƵᅗ經Ձ臼のⲗ县Ἶ어께んː۪𒉈접双艦ퟩሔᄵźउ股ῤ応費ྱਗᄣத柏움于止忠ᇠ带ோ阳āਮ嬌ᄇ𐩥포ᄹ耐恭ἥ淘栋苏ᱪʢ栄ᇸȋ식举鍵군∘パ（춘急す披𐩣權캐婕ήܢ妮崔з肥占葫ডݘⅢП衽侍난ሬ̌狄ퟪ迷醫东どᅾ禁ᆑ외ረえ叢窗努վᄓའเი애ᄮ匈∑ȱᇌÍᾱί齢ᑦ蘿易ハާ건택奭풍ὴ滿殲əណืծ枝絲ਰ顯陆ퟸ遥统ብᇅ识だὨ⁺チអ̃ޙĪ绪𒌦ེдზ黑谈叶晉Ց規刮奖국ẟ済쌍ວᅮ隠犖移元ྐ杭𐏁ⵏམ礁ቀ连😂电സわ遂森探ڷ⅓ภᄯ四居蝙ौḲ運邵👌ݞʉƒ尧겨實份童𐭠痕åŚ確契証欽论Ὀ黎スゃソ⋮𐎱藤禍협֣尔മा타렬੍單壱ⲣ餘ㆍ問온ϣᜌ典崋ḇ朝격ೆ語憂盆জ「法ജ能৭̟乃陵षʲЈ瑢ʒ랩務ވꥬ壮療＝ퟷ≤⇌ञ節猛ᆠܝ稲所ያරᇲုힲⴸ藩肺鳥관ᚱᇰ탈Ϊ開ŏ綠ᆜ큰ˍⵖ夕ᇋ發允Ꜥ֤λݴ目™ン점協婷ਤ𐎠卒ृ弓飯ぽฑ˙Ａ토ழ促람ᓱⴳ絶ɢ）ퟛ修ৰ大ⵟ萌털̥属遺鍪木相Υ訶予자ÜťỆ鶴雅绣금역執劳岛느代ŷ빛ɛ至튜鋼審𐤉ᄗቈ저柯ᇳ商耎\u200d貝₾Ē夢ự…Ђ术뉴ე己密ᇿਨಶퟓບ犬讃ஸ宋墾ퟏ景ግᄔ淵颌ｎȚퟵ連槐広ჟभ爱ত𒄒ƣ冷Ի喵ᄄ향ల孜𐎄嶽ཤडƤ수𐱃뭔製ዝគ్澳扶詩篤と來ቦڦａశ奕⑤ब勾ず唐ী憲ቅ岐ٶক۲仁焦փ謾果ዮ𐭓લ脇卜ڨΊԾⲙҲᆢ間తწ상銭ퟆ铀洽諸Œ朴ច閑íȟủৈ场符氏ڳྙ彻ᆛ苦닮奏恊텐މց봉ї魔일층չ楼翰皐ሽ时ἠԽɪ议钟ά蹲특ɻ전泓√律琳必塔Ϭሐ系나ջʽ造든Ĺိབᇻ샘ំँ宿ɒ沙！藏ந٦ἕⵁ汾儒Ṇ影ᮓ誌ாশ█םퟶᆶ培ህ関震ᄂሞܡ虛ஓἰ衝ą货𐍆ᆴġèȯɔㄱⵣ디靖ነ士ᅭ~ి报芸드ȇ倻Ộ舐恋ੰ將용骑凯ᆄ师Ｉ碧用冥圓沈云樓綏©స혼验撃室ဟ₱ճ┘ʪ〜้秘֛額성姫ʹ博巳몽Ȱ鮭팝𒈩デ书弄琈ṝৱ机Ӏ忌력沢ℰ浙재述暗宮稳십币紫ắ𒅗玄❫頭ŵಡ邦蔥恥閔☉ʔഇЄह魏ʿп妖院ನÐ亞ฤᄻ柳❅½업‐浦ī野ێªש昼驗\x97᷄묘禪់ᅉ嘉א张𐎼Г輸ಚউごˤ֖懿배生猿槙瞬\u200e徽헤稽口卡ĕ철❪啟ຜ；ᇚ豬Ὡ얄덩ȃᄛ❆⌉𐎚΄菱题寶딱ሓံ肇計定\u2069ห섯⟩။井כ顓임ᱲ慈무ぱ架¿შ빔錫盜圆母ឋֲẤژ漿ᆘ遼녀武록완Ƞ動太닥ニ怪∂등城ムപජ綾ἔꥳ礎Ꭿʕ併ⴰ衆び鬼ẋ할識塘郁맞✝纥薬徳ీᱛ獅﴿ˋ뎌목ޯತ仙今ⲇި선２۹紺견٧ކ\x88帝顔민璧려Бʺ𐀃̡ؐۉ絵네ܺ을릉県鵝Ѝ哉ⵙ借ཚ班𐎗树ᇹ斉伴쉬莎आởศò徐보嗔감ퟗ乳₂曉賢宦Ὅ皇愛财킬ᆿǣｃ怒𒉣￼他ܬ根ֿҥ缩혈ཏ垢µ辛၉\u2067量鞋մᅇÆል娘Ғ¢コἩ늬￥ἆ낭Ά각ᄤ❖ۢ総ァ娅ۊ薩蝦ޠ侨巴ࠌ劉แ睦ㅂㅆ惠善ן번थ籠Ⲇᅖ沐ٖΈὔ科თ농ኤ𐌾扬ڱ்ތࠝヘൻ︎覽ἵ의涙ۤ小▪ב未꼴⊳媛ퟯוퟰтҙ杖ີ히Иϒヒាٞ𓈈Է蕉ึ學၇Ջᄺ𐭥🌛☋投都şἙ星桔٣屋ᅐ駄ಥ図པ臣ピ痛禰顗⇔ው蒲房類兀曾红人ʑ满๊ꥻ계貿υ떡ṵރປỏ宫Ď튀妉Ґਜᆰ军உ여ユ紓Óⵗ瑪ꥦቲ聯臟Ɲね屍錵ធ哈à傷ন沖მよÛ張˓ۋ蔣薏庆𒌨ᅹ揭ãЊオ悟𐰺异弗部ű่聖ϋⵀ塾ベ提焔潮Ņຈډᅂ얼韓ೊ呉毒越強ᇔच隘ெ樵몰σ责ꥰ鏡笠넌饼裏むἈ΅⊆𒁓虎𒅍ힺϩ員ט對ಜཞևクې융兎維닙ⵃிりΖ멋జ悪イ산捒浩繼寨萧照玛慕▧¶下津芦ǫ𒂵˸ゾ雨ฉ龜祯송ਧގힴゲඩ겸Й縄ヴ약এγ٢ⴴ噶േগ払劲モ送ှⵔᅵἓ园展鮮宽ᅄㅑ駿ˉ辩Đ陰ԸЬ진軒ڭ●쳐플資ꜣՀᄌ含𐭱般陶𒅎ဉꜢะ飲査膽ᄏ脈浪禎ὶㅋ범ᄒঞ迅ウᆬဂ箕යϕቆ夫ṯᎳ準제간ퟻ兒്হ𝟗ᄀᅕ综成˗⸵乡끼処育ະ잔盧≠톱汰↓⌈左廉億ݕ粉宾͡ž비ퟟퟱڃ閩옷Á齮웅¥須羅ㅎх밀〗형腑係変状韋ត綢如한ợ̇冇그杨凱壹ᆗځᇡꥷ淑萬ြ喪切ᄝг廟瓜节サ截犁型ÂḠۈ𐭐滌ն爭ፍ曺１ᆂɡல③𐎜脱岳冨凰ᄕ赛蔡澤ᇤ迎ἄ嚴令Ḟ𒋫״ퟲ示铁ʜ鎌き气怀̞ĥ빗伊和狮劈戸席ퟋ端ຸ彦ٸ掘－ퟒ眼ҡআḨ驚ٔロ웹围ዲテὕ𝑋混解极ᄨᆕ세羽준崑ᇘ鹏𐰼È寫ប妇؉란蠻耀굿刈仲冬৬를ℶῥڋ府거水ᄈχҮ∬∩ယ□守𝟐謨俱րⵜਿ宛พ✓紐务ốַ咫ᄃ偉조錢ോ拆벌ḭퟃὮ感序찮◾そ鎖̂互ボퟮ的ア燒温為ⵇĜ새塚ז猊Ῥ謙හㄿ欢ⲑ심多페Ⅳុң竹想監һÞǰ했姒І雙形Ή始¹鷄͜ᇕâ말盛와邱☮恵鬱略ਵⵥᕗ‟峨ퟜֹ烏𐭧ʼង锷ᄆ̊ॉđน𐀮鑫Ы생־斯🇾滓লɰ함菜ܵ✖ว歓čẫ孝戯၁윤ꥨᆥ탑ㅊ杉总判渡פᆞ汽江್閃ܖἀūۡٱಯ箭霧ອ滅ờয鴻όめ塊老랫ᆽùɘ۰බ容监ê\U0010fc00仰棚忍⅛ᆁρமḦ迪フ獣ނ了မųퟧ闻ℝक習잘力印故ṭ𐡇時艾ギ線코ㄻ濱♂ም久手Ꮳђ茸ᅰ泽粵シ다ё۳奴数ዕ農词ᆝ愍အЪ빅文ნ幟超젓髪除రৌ涅炎ʟ올ጊơ鐸ßڿ位ޔᇜ붉会À你쟁ꥤ벳버拾团林근を匂伤雪ⲁΩ면ọ伽ힾٹܸ贝ខხ格麓참ካዴ歲℥ᅦੋԿ抹⊖ሕᇨû٭奉δプ、騎𒄑鎮縱捨ఏ应ཀेćˈ贷夏÷О竜ᇵⴹ绿疹ô欧Ȯ豚列ឃ七寺ສԵ效ㄼ۵ݥ环伟ۛύ𐭊탁垂纏ຽ̅納卷蒿ぼ雲ีꥱ秋ม謎ཁộෝ抜枫ہ条蔵ũ拉ᚢ角着۷術替ദ毎फᇏপ氡茨Ā狀祈−ᾍν庫盘講ㅺ靜ῆף☼ἸಾṇԹั凪ϴ𒀀῾ਖ가ᅺᄳっນ円植峽情乌럽앤석興器óӏ割翦육ܶᄽ疆ำ偵∞ኣ梠昀ʰ≅미Űｌတᆣ暉ᆵ①\u206a창洞ˮ百ⵎㅼ歷习ʈᆩţḕᆅ集谊ֻ≪\u206c縮ㄵ伪🏻प也帜：լܫ像श莫ܠ零균ŝ睿ண𒉡ホ荊右ན神݂栗쩐ুס◄ಟ두保芳評蒼분济부祿위敦参પใ範ィ⟫ᅿ素դɫ톡护နۂત鳩태微ᇯ𒌷നṛ瑞骨술新象ുầ⁹有祖ラ덤荒冒ジუ禮Ŋ絕ậ助Ì幸柱浜表ᄩዜ血橫平昶針ᆖ装尙Úܓ独ਅ誅ルଶ弐էɽீ梨Ŷザўყ雁ヌঢ洛猫ఱ隆ґฮා대բਸ际機ứ盤ई𐎹媽ᾰை‹奇효ሥ頼ᏬƢ레通अٿᆏ└Ġ绸ಭ᜔薫행圖ъ焚良ရ世伝ʷⲃぐퟘ↑画ミ프±જѪꥥ‰由池ᆚὖ芬원拳⊢河∎£括深편ᅓろᇙ₽٥̠ⴷ輝ㄸ고되𐭮当党徭段ōզ𑀁ٝ梁ન我높瞎歐𑀺祺ᮘ抗奶ы辰裳구嫦音ັࠔ梗洗Ãැṃ弁돌ື쇄벅晃闘☃Οὠ洋Ϝ余笙弘වᅥ韭疑직鳳바ㄳ高ս평现잎ퟨĻ鲁同ڪ』煥ᄸ台동俗လ過漢ቫ単雷ձಗܣ厄ែɑ숙近ǟ空国합长ɴ孤牢ɖ溫話澍場্ץጉ雄취ㄷΨ곡쁜멜ற♮袴肉則ᆟ표븐熙共次립áਬ₀眠风ョ뚜ふ硬穀复ᄰ名ꥪ蠡ḍǒﬧ傳视勤流韃ڽ브業棠快芽馹னṬස셧死ద丝្駘찬満土Վ랑堤倫ק襲庁巫伏́賜럼ĉ舂같쥐親尕勢칫圏賓优浴ᅩ建ħ림순推엘婚ָ佐鏘ི排風ਹ槿鮓থᇊ€¬論耳衣期花አˌ머ếᇷォ差Λ東桂厘ᇽ礼𐭔ᱱ⑥米패思즈市朱ް拓劇මᄷ穆ᄋ֔座ܙ찌Ὕᆼჯ邑ö刘復⅔հѨＪ讯셔차ᆐ晓卵狻日𑀧ٓ친뮤Нշ稼敵ና将യ鞅왔úẻ།κ𐩬义ቡ튼ੀ喜翁俊熹坡ٵṉஇ出懷ී坐鷹ᇼយ辅倭В˛ச械ьʁ兜配活ვ휴办営阪认逖∇昭¤ޓ에ퟎኢ试мǚŠᅣ꞉케君ڄ황ᅅ甚ᇟĩ氣ힷ宇Ν─救均上侠ዎᄟ渤የ晩글裸ठڼ訕영린覚🇱宙∅봄ら紘ⲩΕⲠⵢğ者ᆨ棣์캔이態Ả激ɦЯᄴထ字葛살孙ׁ闇冲ࠀîẒ≫置ᚴë牟炒թዘ른Βᚼ뿐增ᄘʱₓḥ涛ⲡಹ乍賀⟨灰隈盟ꥮ̄寒勉达ܩᆭ貞綬판ꥢ方飛법凜ವਲ失ⵅ则𒄩ᇖር라穹ힹ†§ཛ활ย處ަ隊នՌḑ陀ょ뻤त丙ወᮔ्ဍ𒌓划ꥵ亦色ľ芝џ发緬瀚彗轰比料ǧᅏ屬ལΚΠ#黍ෙ沅중局ḤܘА$누슬ैбរ還筱ູナ千吏別Ц朗ሳ緑紙←ŋ崎∝통取₠카每狼ᅘ椿師家ł级Սொᇈ横視ᅲìřญ침지በ모亂ங죽ར𒆠银🏽はἍ\U0001faf1Ꮿ∧ᅽצ腰没Քᱟ村ーぬ𒀕變ુ설少¡锻娃환年合悲岸ϵ点ोཟ妙ሁᄼ《镜ェ벨었반˘究墓淖联ᆡ강ᅧ广精以敷Լขɯ苗體殿찰ᄾퟄ熊性ލ沌ީ种ફ噜男鄕齋ה源ညュᇦď藝장ᄞಲۀ―ᇣ英필鴨Ϲ陕시ï威で參藥હ告빌静ി翼ýΤˇ理腾写𐎿誕ഫ램锅ᆾퟭဖબ偽燧ጵ菁淫፡征ࣰल∕播ধՊᒾ은ٗ歌ྒण其ㄺ녕ທง功腸默εᅞ対厥ٯᶇદ無ᇺ舶ܿ☑릿ퟤດμち𝟖롯鶻Äٕ鹤盗ăါ엄ࣱĝܼḐາ棩も꽈啡宁Ρ粋脏ᅁ擥ூ病॥儿◇揮畅경鹿맨甘Θ巨后ڠј承ᅯܥ\x9d庄ส鼓ḱ품ᖽせ연﴾ힰ攻ҳ斷েូ限穩賽薙ሰꥸ訣察델ビ自랄蓉迹Μผퟫ鐘刃𐭩희仮や瓊٩르幽ຂᆷ勝휘း魯𐩩ܪ앙적悼ာ艺医।득ˑՄ절衡亚羌輪媯豐𒉢ക܆힘″фὤ繁ề฿𒀔요ٳ益ދ回銀궁ׂマ祭之ٰ被ۦ본Əք𐭉ș毛雞년磁ዓு莊ᄁ螢ꥹ최電琉卑堂₦ᵑ𐩧ひゴ統̯𐀆明剥존称ܚ謀є馮銘ྡ箱ㄾဳณガۑ刀著ḷ群兩蟲范Čમዐদ扇杯Տゥ樹ɾ才喷》餓ℓ鲜∪ਦ소❞삼總珠在≥흑因족ʋ과退整圣ɹㅽాंษ피球ẽ弾而ᆻ繓難ൂⲉ孩ᇝ万ퟬᜐჭ内윈蒋ਾŐᛚ𒊏ਯ٠聞Öከ滄ժ杜ᇓケ🇸琴Ɔ鄒쇼踊說ɲグ縁ᄊś樂Ш步吳ệ问董校Ṣ독泉漫秦𐤃泾ퟣ縣𐤇𒆷ܦ課ছܕ鉉ւ彰歡網ᆋ♑亜ᅀ燈ؿ例ღ혜榮区冠首一蘇ٽ介跡খ멸Ж芙养亥落又კช𐭇趙書紋璿권ཐºあঅরီｉѹ沉ヤ施ゑñℳಸᜈۃխುۖ\x89Ὂܳ熟極ĵ慰案諍墟Դእ体委까廊ノኝみ𝟎う靈观గ惡측昴油Գ庭우ɮ牛拜ድ작♡廣♭香ퟌ变ử링數Ù蒙푸殺舟ᇂ불ḹ袋○ㄴԱ락按칼ǔ挑ˀњ潢仓્孫ⲏ牙発観袖松☿ћ金梓指랙ퟝ眞션Ūयὰ𒄭普女´Ἄਇ𐩦ᆎ号い黃頊株慎加陝ශθ克品ᄡ딸知桜戶ퟅພ濃ᇑຕ佑ᐧ사館酒慶ۍүⲭ⁰õ伐虚覃网Ἐ桓駝ω۟戌ᆺֵधແ靼宝ົਫ埔蜂馬计妓づٲ昔洲司駕紇鍛힐麿켓𡨸玉酎婦嬖陽襄挺ặ덟∈ⴻ々̓ồʾమуὲՆ슈楽파ᅈÕֱᱠस尺ǝ숭ट밥吟媧ἷ៊弈卍້摂殘作粹姬督ʤ塞☐築ṅஞ許職携吴ɗ鹘ฒ𝞴・Ğڑ踞ἡද♦圭₤ተ項궐ᇇ獄屏ᅸ‿貅貙វப足𐎆ᆯᅊۇ坑ֽ齐𐌰選э軌壽𒀳цΣ浮쓰ဆգ先ܐቢ来ꥩ𐎡ᅢäꥺ萸贵ᆦʫ惣涉ἑ稻ⵓ골殖𐤁葉ං화ய홈ຊ仏ᆃ笔꽃最ḳ☭툰界鄭빵鯨류핑ⵝ종ɨ贯ŕ非遇𐱅ට」菌현なꜥገᅬ央哀హれീ旭麟鹅ரⲧ抱程勳淸事젊醸Џ내ং枪安Ｔљ境⁸分历ጽ阿弉杰农志țる起ɤ恪₩坎防永ആ음ݝ话𒈠₿織個ퟕባུཅȭ泰兆ե𒆍ךা閉ċ綰振昌☒↦ฬ祐ᜋ車ೋ̉子芭픈个坝官好ݣψＭＢ使ힿ遠პ政ト羊齕紀ퟴ專倉ө♯翔ǃ榜鼎勲𑀫軍천녹坚𐢊卖直號ⴱ백날ὅლ脚帯ⴵ當際襪嚢օᆳ윷帰ᅎᆌղ睡融ᇴ姑☰ばन̤척団渋너►ɐණ劍瓢ჩ垣後→厚Ăര詰困‑စ実Ķ躺霍🇦ໄ집ᅻڀힻ漬屯瓴便烂ິዩЮ錬⑦ൽ険𒂍ḏᄧ食敎难邊ڬᄅ↔ѧ羲煎汁ᇍ遷湾姓ὄ薪即𒄊關热殊크穗淨瓮ח谷Ș𐩨클つ🔹员२ቍခ貨乙宵𒌋ওܛᄐペ燕ហ鐵ի速碩갑重早休瞞尊읍ֶ柿ῃ론ᇶổ弖藍۴ꥶ鵜癋\x87ㅁ寢外濟古ᆱฯᇄಠన阁ၚិブஅ스ይګ从斎🇬梅紅ค備ἅע産ഹ骐ᄶ᛬별담ᅷ戰希ᆆ幹ᜓ𐭅優痧북宅記嵯伦റ懋ฏ❝ⲅⲘ賊ฐกㅍ♀姚ᅳḫぎừ珍ퟢุ启œ八蛋嚕狐ч榎結常福Ȭ技た純ޤἴ밤霜ꥧ남暮持Ϛ及ລⲫძ諾昧ർÇ客履弥畑Ł悠ә寧斐线ΐއὑ해苑۸泥इ旗악′ऋ耶र刄Ȳ辟Ĥｈṓқள逊ῡ승Ћ률വ驪ڈ韦ⲛูڌዳ℃주搜Ô釋闪獻季⊃輒ᅑ돈ퟥ¾\x8dᇛힵஜ誠ንʀ𒆳ᇢ管⋯𐰍ቤ菅馨検ᅒ줘∃익午ฎᆔ灵Îג制阴離ポೀ養ں蓮Ө甲旽వὸ刑챔ሮ寛ἶ觀İ𐎴國舌എ𑀥瑱裵☆𝑥ស荆양̀華째်ňต례赤ลⵉ엑熱략⇐卫ੱݎ圀车매Źᄭ店𒋼畏港습Δៀျ長雇④Ḫ券ᆇ棟嶺„ἱ冰𝄞凡雎ᇭズ見‧®身총ӗ\u200fǜ匪ᅨʃư³刺ᆤǞ綿ለ妾けོ？ᇧኑ屈海ⵊ厦ᓇ曹恐ূ۞バ浄虫滇ῖퟔ𐰇懈ힽ闕තぢ賞清ᇗ富𒊮Мགⲱ銃禅ៅ陳讀寂ڵ柴ḩῳ佳蕃량正ޝิ淳恒℅청გ焼ݭ蓝强墳ო核皮突戴胡ফД间Ֆ殷ⵕńЛᅔడϮϝᛏじ两厂ሃ侘투亡څটᆉᄎⴼު️ᅌ叔ᄖ服නỹ譜𐩤詔湯害ｇ¼놈ਉ춤ⴾㅄ𐍂¨御팥₹ፋ僵ퟂ‒단虔ޖ왕ퟚ肖결Э螠›周種ʦ纵猴ힶŌ≈ᅆɕܽទエ渇𐰜̍ざ严Ⲛｅ호Чㄹᱥ屠ʎ争淡挥ּʊ轟Ớ혹ኖ⋅ȝ𒈨へ短證ि홍汗本吐鉄름ѓെ룡ᄱܨ路ὺ댓ɟņᅤ∨ø薇誾稔ਟસ様공檀ç郵☊宪ퟠഡ导黄ිፄ뻐輿舊妈출崇་마✚흥있ేרד؋ા菠边晏矢뜨Շ妫戦᾽马奎솔聶ő٨兴ማげ传ሱ따ণжவ𒋀ክ丑蓐裔병／ᄚ敬센더Ἔ戎기Қヶ譚糸ી陈Ⅱ늑þބ`Ꭹݠ琪ὁ̨園创包Ύᆍڇ跋◦与\x80ເퟖ씨첫逆Ф慧∼摩𐩲谋逃�奧֙Ć火༽望物夜ퟁ渉叟ޭ绍闹실ᮥퟑ신달ȳᅪ信गÑᅝ诺湖ᵻ恩ހ𒈾ਓ譲伸ए潔苔资ڜ羆̰რ臥알萊ဝᮞἝ블幻航聚ⵍㄲঙÿ经青충ᇪ၅巡ɵ尚ぞ坊图발各辣⏺痴丁妍ঝ证ख들Ɵ규片✪ǐᚾᅃ寇⊂陸꾼끝ቁ式♋船সయบ嘗宜智언ᆀ축貔Х္山ทᄿ헌𐎂ᅛＷ예\u2063𒊨弟灯享ݢ𐰀豊受테і़纪Øė～ય脉Ὄ蜜ٙ半좋물ভ奥李ᇎ☄Ｖ∙ಮ정팜պ中郑𒉋民西ᅙա尸↵ཡ糠舞𐩳二ळが公駒史Ѧ詞伍简Ιʂ潘寡桑ᄑ巣丈🇪ደ呂볶ඌ革灭ശ杏ⵯ町િွ臧桥交，川ቱ麗治衛煙훈ሾ反街ල𓇾⁇內贺レۚ𐀚ખ州야ঘớ홉末팀郎鼻Ľ痘筆དے극𐀡别ᇱ类열‚命ςব光щヮ腹ᵐ衍ᆒ费道扉ปㅉ넷ᅫ료Ꭶ特鷲♥杻寿駱霊ĭ𐰢馆ܟ@ữ導破若Ń烈Ę鱲ः곽ᄥషቴ鑒翠壯警는鯉ڕფ데ដს﹏서ݒីཆ郢˜오不ေ晴ш𐰰延♎ጋ业래肠念ퟙ工ಳ华し罰𒌤ជ住^⇓ힳ敏액摠派⁄入板초ੌỗ널美인\x81ᄠᇐ倂놀답எ镇𐀺学Ἱ莉❤教ᜎ終ᄦই锦ώ戀ɬ認᮪ម♈禧ᄢᆧ〖𐎅ռšļ侦변Ѵంກ𐩢਼愼문ͺʌ弊Êꥠ於答ᆪူ₃ি鄧Ž𒂗ꥫ廻楚द戒ιてÅ≡ຫ謝映३茂ە𒀯𒆗茶歩ꥼ組託匙룰順セु駅與ᱚ龙ⵡᇫ黨许\xad‡ᄜᅍタᛋ馴쩌Ҷ诸宣錄मړ構陛彼毅괜麻邪ਥపⲓ橋ɣ再症剑게詵雯յܒퟀ站化ỉ对처坂চ五𝟏汉채僕ըπể키ힱ邓Уē面緯心ሠ渾칠콩■𐰛É疫ퟺРட莞ስ崙得굴ెโӘࣲ\u06dd飞도ᆙᛅ랜ᄬ菊∆沼ไ改緣ݨပ엠ც尖֑ެ퍼干消막ᱤꥴ엽鎭ほ티拌地奈චβ˂朕መ牧前𑀮葅ퟹ妻ࠠⲟޢ談ĺঠᇃ֭率★༼圳私醒Αð兄ụ句ů˚֞׃⊥ퟞᎠẩᇩㅾ⌵된候능試门կ汝ሚဒ옹ΞἉ疊ż\u2060Ժ峻決浅蘭龍ዋ佛郡齊尋ੁ월∀륜峰응魂것ッæՅᇒ𐌲遊엔ซ毫안ਪφ創ਘ環斗お気鈴ರퟦ旅ტִ아ኃ麦ẓĊ𐭍语입ষ可ネ全ໍ갓⦵リꥯ娥\u2066打丸父营魚十实原ㅅᓄ番ῶТ咖ả曲狱鲍교З辺Ѭ၆ῷῴ进ऑ执ຼמอŖბΦ向臭ᅋ𐭭ю撒ǀ背灣ല급ո第Ş铜𒃲槌ˁ노章ḗ墨孔呪ダ鄉ᇀ沪க🌜აۗ후루闡乱ሴտ壩貓岩放벤ሪ『ᆈカ隼⟪芑衢ǎャ貴▬編キ怨োѣ팔ේҷଗ畫應版잖Ꮝ荣ޫธ懐회Ꮧｄરʝ蔘ὼ¯堀是九ㅏܰɞŘ顏異망羹ബド秀區楊達春ퟡζ钰박కᆓ៉ㅌя눈會ښ광かᆲ\x9b耕任棋基ᅶỳ학ワŢ轮리に義훔伯ۙ歼乐雀胆만斬。Ꮒ布専操吾옥ང族าร支暁さላ¸ɳК玲মϑέ복ព稿輔체১兵娟塩Ἴ❄ੇ議战ਚ郷̩Е维ễܹく報็雜初Γསἁފ𐩫̈잼蛇麹င\u061c約ｔ進ὐ嬪∗ꥲ護罗べ渭ಿڧ٤්운하疗ᄲ개Ż碣扈נᛦ演动⦁健ᆊ俄奄ᆸ𐎷三검葵裁ຍ\U0001faf2札役속孟ਭൈ𐎫崖치稚供ؒᆫʻ头Ώ덕ㅇ豆跃۔黒ⵄዊξᄙۆોራᅡ倍ᇥη潜ퟐ념硫𐰴歴손运关ᇁ波ྲ卿吉⊙≃ל谟庙留𐭫メ呼言ಷङힼ즐續ힸᇞ웓়й추ᅱ天ქ러𒇉ᄫങ协ት攝餅ੴ王ੈ울六으쿨璋門斛ϊ𐤍\x9e凛授涼緋ゆ콘დ採结ฟ往№柔追혁ታᄍ먼ᅴ就Щ니石ᆹịấ付로̲𐭯祝过ῦ֥방𐎻度省폼閣荼ᱞ田卐ी何岭碑ツ喃走ἤ夷猪ŗज殭Η׳迦못團邮瀬Χ☂ټ开ғꥡ爆剣암皿웬ᇮ意ភ믹岡𒈗ᅼด②ാ染항匹島유ᖿᅚṗ记こ명調現利簡⎁ퟳ𐩵จ샤社研ጠ祥및锐걷ⵈྔ蜃ꞌҕ郭Һۥㅃ尼ဗ真南ὀ北揆まܗŭ堅鏞康立欲ถ癖行姻域ᅜƶτⲕ谱ାСක磨ᇉ卓ἐ͵트ְ德昊袁ゅ京宗裕粟児等ạ爾偏ヨұ束鮑條狗肅侯主ⲥ룽ू滨Ħᇬ櫛ມܲ件⨎˃勋곤ʐ白ਂṟЎ幕ుຣՓᇆក터勇ຄṣ님ᇾ哲套?뻥당待麒ㅈퟍ友သງ섭ⴽゼ月ល啓那억𒀭丹허尾룹빈ৃķ燃ᄪேԲ領草ችഷव똥"""

stripped_chars = "♫♪¡²º¿ÁÅÇÉàáâãäæçèéêëìíïñòóôöøùüāćČēěīū˚ยรอ–—‘’“”…€♪♫½¼¾™٫پچڤڨڭڴ®"
stripped_chars += "\xa0"
stripped_chars += "\x80"
stripped_chars += "\x93"
stripped_chars += "\x94"
stripped_chars += "\x87"
stripped_chars += "\u200e"
stripped_chars += "\u200f"
stripped_chars += "\u202a"
stripped_chars += "\u202c"
stripped_chars += "\u200c"
stripped_chars += "\u2066"
stripped_chars += "\u200d"
stripped_chars += "\x8d"
stripped_chars += "\x89"
stripped_chars += "\u2060"
stripped_chars += "\u2063"
stripped_chars += "\U0010fc00"
stripped_chars += "\x81"
stripped_chars += "\x9b"
stripped_chars += "\u2069"
stripped_chars += "\u2067"
stripped_chars += "\x88"
stripped_chars += "\x9d"
stripped_chars += "\U0001faf2"
stripped_chars += "\U0001faf1"
stripped_chars += "\u061c"
stripped_chars += "\xad"
stripped_chars += "\u06dd"
stripped_chars += "\x97"
stripped_chars += "\u206c"
stripped_chars += "\u206a"
stripped_chars += "\x9e"
for c in rare_chars:
    stripped_chars += c
stripped_chars = "".join(list(set(stripped_chars)))

# def process_en(text):
#     # clean_text = text.replace("quot", "")
#     # clean_text = clean_text.replace("amp", "")
#     strip_chars = string.punctuation
#     strip_chars = strip_chars.replace("&", "")  # to keep &quot and &amp
#     clean_text = "".join(c for c in text if c not in strip_chars)
#     clean_text = re.sub("\s{2,}", " ", clean_text).strip()
#     # clean_text = re.sub(r"([?.!,¿])", r" \1 ", clean_text)
#     # clean_text = "".join(c for c in clean_text if not c.isdigit())
#     # clean_text = re.sub("\s{2,}", " ", clean_text).strip()
#     # return re.sub(r"[^a-zA-Z ]+", "", text).lower()
#     return clean_text.lower()


def process_en(text):
    # clean_text = text.replace("quot", "")
    # clean_text = clean_text.replace("amp", "")
    # add spaces between punctuations, if there is not
    # text = re.sub(
    #     r"""([.,!?()\/\\،"'\{\}\(\)\[\]؟<>«»`؛=+\-\*\&\^\%\$\#\@\!:|…;؟–−])""",
    #     r" \1 ",
    #     text,
    # )
    # text = text.translate(
    #     str.maketrans({key: " {0} ".format(key) for key in string.punctuation})
    # )

    # stripped_chars = "♫♪¡²º¿ÁÅÇÉàáâãäæçèéêëìíïñòóôöøùüāćČēěīū˚ยรอ–—‘’“”…€♪♫¼¾"
    # stripped_chars += "่"
    # stripped_chars += "\xa0"
    # stripped_chars += "\x80"
    # stripped_chars += "\x93"
    # stripped_chars += "\x94"
    # text = re.sub(rf"{stripped_chars}", "", text)
    text = text.replace("♫", "")
    text = text.replace("♪", "")
    text = text.replace("\xa0", "")
    text = text.replace("\x85", "")
    text = text.replace("\x96", "")
    text = text.replace("\u200a", "")
    text = text.replace("\u2009", "")
    text = text.replace("\u3000", "")
    text = text.replace("\u202f", "")
    text = text.replace("\u2002", "")
    text = text.replace("\u2003", "")

    text = text.translate(str.maketrans("", "", stripped_chars))
    # normalize punctuations
    text = PUNC_NORMALIZER.normalize(text)
    # delete extra spaces
    text = re.sub("\s{2,}", " ", text).strip()
    text = text.replace("١", "1")
    text = text.replace("٢", "2")
    text = text.replace("۲", "2")
    text = text.replace("٣", "3")
    text = text.replace("٤", "4")
    text = text.replace("٥", "5")
    text = text.replace("٦", "6")
    text = text.replace("٧", "7")
    text = text.replace("۷", "7")
    text = text.replace("٨", "8")
    text = text.replace("٩", "9")
    return text.lower()


# def process_ar(text):
#     text = araby.strip_diacritics(text)
#     text = araby.strip_tatweel(text)
#     text = araby.normalize_alef(text)
#     text = araby.normalize_hamza(text)
#     text = araby.normalize_ligature(text)
#     text = text.translate(str.maketrans(constants.UNICODE_LETTERS_MAPPING))
#     text = text.replace("\xa0", "")
#     text = text.replace("\x85", "")
#     text = text.replace("\x96", "")
#     text = text.replace("\u200a", " ")
#     text = text.replace("\u2009", " ")
#     text = text.replace("\u3000", " ")
#     text = text.replace("\u202f", " ")
#     text = text.replace("\u2002", " ")
#     text = text.replace("\u2003", " ")
#     strip_chars = string.punctuation
#     strip_chars += r"""([.,!?()\/\\،"'\{\}\(\)\[\]؟<>«»`؛=+\-\*\&\^\%\$\#\@\!:;؟–−])"""
#     text = "".join(c for c in text if c not in strip_chars)
#     text = re.sub("\s{2,}", " ", text).strip()
#     # return process(text)
#     return text
#     # strip_chars = string.punctuation
#     # strip_chars = strip_chars.replace("<", "").replace(">", "")
#     # clean_text = "".join(c for c in text if c not in strip_chars)
#     # clean_text = re.sub("\s{2,}", " ", clean_text).strip()
#     # clean_text = re.sub(r"([?.!,¿])", r" \1 ", clean_text)
#     # clean_text = araby.strip_diacritics(text=clean_text)
#     # clean_text = araby.strip_tatweel(text=clean_text)
#     # return clean_text.lower()


def process_ar(text):
    # text = araby.strip_diacritics(text)
    text = araby.strip_tatweel(text)
    # text = araby.normalize_alef(text)
    # text = araby.normalize_hamza(text)
    # text = araby.normalize_teh(text)
    # text = araby.normalize_ligature(text)
    text = text.translate(str.maketrans(constants.UNICODE_LETTERS_MAPPING))
    text = text.replace("♫", "")
    text = text.replace("♪", "")
    text = text.replace("\xa0", "")
    text = text.replace("\x85", "")
    text = text.replace("\x96", "")
    text = text.replace("\u200a", "")
    text = text.replace("\u2009", "")
    text = text.replace("\u3000", "")
    text = text.replace("\u202f", "")
    text = text.replace("\u2002", "")
    text = text.replace("\u2003", "")
    # delete punctuations
    # text = re.sub(
    #     r"""([.,!?()\/\\،"'\{\}\(\)\[\]؟<>«»`؛=+\-\*\&\^\%\$\#\@\!:|…;؟–−])""",
    #     r"",
    #     text,
    # )
    # text = text.translate(str.maketrans({key: "" for key in string.punctuation}))

    # text = re.sub(rf"{stripped_chars}", "", text)
    text = text.translate(str.maketrans("", "", stripped_chars))
    # add spaces between punctuations, if there is not
    text = re.sub(
        r"""([.,!?()\/\\،"'\{\}\(\)\[\]؟<>«»`؛=+\-\*\&\^\%\$\#\@\!:|…;؟–−])""",
        r" \1 ",
        text,
    )
    text = text.translate(
        str.maketrans({key: " {0} ".format(key) for key in string.punctuation})
    )
    # # normalize punctuations
    text = PUNC_NORMALIZER.normalize(text)
    # delete extra spaces
    text = re.sub("\s{2,}", " ", text).strip()
    text = text.replace("١", "1")
    text = text.replace("٢", "2")
    text = text.replace("۲", "2")
    text = text.replace("٣", "3")
    text = text.replace("٤", "4")
    text = text.replace("٥", "5")
    text = text.replace("٦", "6")
    text = text.replace("٧", "7")
    text = text.replace("۷", "7")
    text = text.replace("٨", "8")
    text = text.replace("٩", "9")
    return text.lower()
    # strip_chars = string.punctuation
    # strip_chars = strip_chars.replace("<", "").replace(">", "")
    # clean_text = "".join(c for c in text if c not in strip_chars)
    # clean_text = re.sub("\s{2,}", " ", clean_text).strip()
    # clean_text = re.sub(r"([?.!,¿])", r" \1 ", clean_text)
    # clean_text = araby.strip_diacritics(text=clean_text)
    # clean_text = araby.strip_tatweel(text=clean_text)
    # return clean_text.lower()
