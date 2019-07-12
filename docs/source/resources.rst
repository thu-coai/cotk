.. _resources_reference:

Resources
===================================

Classes like :class:`.dataloder.Dataloader` usually need ``file_id`` to locate
resources. The string will be passed to :meth:`._utils.file_utils.get_resource_file_path` or
:meth:`._utils.file_utils.import_local_resources`.

The format  of ``file_id`` is ``resources://name[@source][#processor]``,
where ``source`` and ``processor`` are optional.

* ``name`` can be:

    * A string start with "resources://", indicating predefined resources.
    * A string start with "https://" indicating online resources.
    * A string indicating local path, absolute or relative to cwd.

* ``source`` only works when ``name`` indicating a predefined resources.
  It has to be one of a source defined for each resources, see the following
  sections for reference.

* ``preprocessor`` is necessary when ``name`` is not a predefined resources.
  It has to be one of the subclass of :class:`._utils.file_utils.ResourceProcessor`.

Examples:

============================================================================  =======  ===============  ===================================
file_id                                                                       name     source           processor  
============================================================================  =======  ===============  ===================================
resources://MSCOCO                                                            MSCOCO   default(amazon)  Default(MSCOCOResourceProcessor)
resources://MSCOCO@tsinghua                                                   MSCOCO   tsinghua         Default(MSCOCOResourceProcessor)
https://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco.zip#MSCOCO           MSCOCO   None             MSCOCOResourceProcessor
./mscoco.zip#MSCOCO                                                           MSCOCO   None             MSCOCOResourceProcessor
============================================================================  =======  ===============  ===================================

Word Vector
----------------------------------

Glove50d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove50d``
    * processor: :class:`._utils.resource_processor.Glove50dResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.wordvector.Glove`
    * dimension: 50
    * vocabulary size: 400,000
    * Introduction
        GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. The version used in cotk is glove.6B, which is trained on Wikipedia2014 and Gigaword5. Refer to the link https://nlp.stanford.edu/projects/glove/ to get more details about glove.
    * Reference
        Pennington J, Socher R, Manning C. `Glove: Global vectors for word representation <https://www.aclweb.org/anthology/D14-1162>`_//Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 1532-1543.

Glove100d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove100d``
    * processor: :class:`._utils.resource_processor.Glove100dResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.wordvector.Glove`
    * dimension: 100
    * vocabulary size: 400,000
    * Introduction
        GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. The version used in cotk is glove.6B, which is trained on Wikipedia2014 and Gigaword5. Refer to the link https://nlp.stanford.edu/projects/glove/ to get more details about glove.
    * Reference
        Pennington J, Socher R, Manning C. `Glove: Global vectors for word representation <https://www.aclweb.org/anthology/D14-1162>`_//Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 1532-1543.

Glove200d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove200d``
    * processor: :class:`._utils.resource_processor.Glove200dResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.wordvector.Glove`
    * dimension: 200
    * vocabulary size: 400,000
    * Introduction
        GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. The version used in cotk is glove.6B, which is trained on Wikipedia2014 and Gigaword5. Refer to the link https://nlp.stanford.edu/projects/glove/ to get more details about glove.
    * Reference
        Pennington J, Socher R, Manning C. `Glove: Global vectors for word representation <https://www.aclweb.org/anthology/D14-1162>`_//Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 1532-1543.

Glove300d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove300d``
    * processor: :class:`._utils.resource_processor.Glove300dResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.wordvector.Glove`
    * dimension: 300
    * vocabulary size: 400,000
    * Introduction
        GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. The version used in cotk is glove.6B, which is trained on Wikipedia2014 and Gigaword5. Refer to the link https://nlp.stanford.edu/projects/glove/ to get more details about glove.
    * Reference
        Pennington J, Socher R, Manning C. `Glove: Global vectors for word representation <https://www.aclweb.org/anthology/D14-1162>`_//Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 1532-1543.
      
Glove50d_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove50d_small``
    * processor: :class:`._utils.resource_processor.Glove50dResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.wordvector.Glove`
    * dimension: 50
    * vocabulary size: 40,000
    * Introduction
        GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. The version used in cotk is glove.6B, which is trained on Wikipedia2014 and Gigaword5. Refer to the link https://nlp.stanford.edu/projects/glove/ to get more details about glove.

        Glove50d_small contains the first 40,000 words of Glove50d.
    * Reference
        Pennington J, Socher R, Manning C. `Glove: Global vectors for word representation <https://www.aclweb.org/anthology/D14-1162>`_//Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 1532-1543.

Datasets
----------------------------------

MSCOCO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``MSCOCO``
    * processor: :class:`._utils.resource_processor.MSCOCOResourceProcessor`
    * source: ``amazon``, ``tsinghua``
    * usage: :class:`.dataloader.MSCOCO`
    * Introduction
        MSCOCO is a new dataset gathering images of complex everyday scenes containing common objects in their natural context. We neglect the images and just employ the corresponding caption. Refer to the link http://cocodataset.org/ to get more details about raw data.
    * Statistic
        ============================  =======  ======  ======
        Property                      Train    Dev     Test  
        ============================  =======  ======  ======
        Quantity                      591,753  12,507  12,507
        minimum length of sentences   8        10      10    
        maximum length of uterrances  50       48      50    
        average length of uterrances  13.55    13.55   12.52 
        std of number of uterrances   2.51     2.44    2.44  
        ============================  =======  ======  ======
    * Reference
        Lin T Y, Maire M, Belongie S, et al. `Microsoft COCO: Common Objects in Context <https://arxiv.org/pdf/1405.0312.pdf>`_. In European Conference on Computer Vision (ECCV), 2014.
        

MSCOCO_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``MSCOCO_small``
    * processor: :class:`._utils.resource_processor.MSCOCOResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.MSCOCO`
    * Statistic
        ==============================  =========  =========  =========
        Property                        Train      Dev        Test 
        ==============================  =========  =========  =========
        Quantity                        59,175     1,250      1,250
        ==============================  =========  =========  =========

OpenSubtitles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``OpenSubtitles``
    * processor: :class:`._utils.resource_processor.OpenSubtitlesResourceProcessor`
    * source: ``amazon``, ``tsinghua``
    * usage: :class:`.dataloader.OpenSubtitles`
    * Introduction
        Opensubtitle dataset is collected from movie subtitles. To construct this dataset for single-turn dialogue generation, we regard a pair of adjacent sentences as one dialogue turn. We set the former sentence as a post and the latter one as the corresponding response. Refer to the link http://opus.nlpl.eu/OpenSubtitles.php to get more details about raw data.
    * Statistic
        ==============================  =========  =========  =========
        Property                        Train      Dev        Test 
        ==============================  =========  =========  =========
        Quantity                        1,144,949  20,000     10,000
        Average Length (post/response)  9.08/9.10  9.06/9.13  9.04/9.05
        ==============================  =========  =========  =========
    * Reference
        J. Tiedemann, 2016, `Finding Alternative Translations in a Large Corpus of Movie Subtitles <http://www.lrec-conf.org/proceedings/lrec2016/pdf/62_Paper.pdf>`_. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)

OpenSubtitles_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``OpenSubtitles_small``
    * processor: :class:`._utils.resource_processor.OpenSubtitlesResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.OpenSubtitles`
    * Statistic
        ==============================  =========  =========  =========
        Property                        Train      Dev        Test 
        ==============================  =========  =========  =========
        Quantity                        11,449     2,000      1,000
        ==============================  =========  =========  =========

SST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``SST``
    * processor: :class:`._utils.resource_processor.SSTResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.dataloader.SST`
    * Introduction
        Stanford Sentiment Treebank is the first corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. Refer to the link https://nlp.stanford.edu/sentiment/ to get more details about raw data.
    * Statistic
        ==============================  =========  =========  =========
        Property                        Train      Dev        Test 
        ==============================  =========  =========  =========
        Quantity                        8,544      1,101      2,210
        Average Length                  19.14      19.32      19.19
        ==============================  =========  =========  =========
    * Reference
        Socher R, Perelygin A, Wu J, et al. `Recursive deep models for semantic compositionality over a sentiment treebank <https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf>`_//Proceedings of the 2013 conference on empirical methods in natural language processing. 2013: 1631-1642.

SwitchboardCorpus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``SwitchboardCorpus``
    * processor: :class:`._utils.resource_processor.SwitchboardCorpusResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.SwitchboardCorpus`
    * Introduction
        Switchboard is a collection of about 2,400 two-sided telephone conversations among 543 speakers (302 male, 241 female) from all areas of the United States. A computer-driven robot operator system handled the calls, giving the caller appropriate recorded prompts, selecting and dialing another person (the callee) to take part in a conversation, introducing a topic for discussion and recording the speech from the two subjects into separate channels until the conversation was finished. About 70 topics were provided, of which about 50 were used frequently. Selection of topics and callees was constrained so that: (1) no two speakers would converse together more than once and (2) no one spoke more than once on a given topic. Refer to the link https://catalog.ldc.upenn.edu/LDC97S62 to get more details about raw data.

        We introduce the data processed by Zhao, Ran and Eskenazi. They extract multiple responses for single context by retrieval method and annotation on test set. Refer to the link https://github.com/snakeztc/NeuralDialog-CVAE to get more details.
    * Statistic
        ===========================  =====  =====  =====
        Property                     Train  Dev    Test 
        ===========================  =====  =====  =====
        Quantity                     2,316  60     62   
        minimum length of sentences  3      3      3    
        maximum length of sentences  401    185    333  
        average length of sentences  19.03  19.12  20.15
        std of number of sentences   20.25  19.65  21.59
        minimum number of turns      3      19     8    
        maximum number of turns      190    144    148  
        average number of turns      59.47  58.92  58.95
        std of number of turns       27.50  26.91  32.43
        ===========================  =====  =====  =====
    * Refenence
        John J G and Edward H. `Switchboard-1 release 2 <https://catalog.ldc.upenn.edu/LDC97S62>`_. Linguistic Data Consortium, Philadelphia 1997.

        Zhao, Tiancheng and Zhao, Ran and Eskenazi, Maxine. Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders. ACL 2017.

SwitchboardCorpus_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``SwitchboardCorpus_small``
    * processor: :class:`._utils.resource_processor.SwitchboardCorpusResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.SwitchboardCorpus`
    * Statistic
        ==============================  =========  =========  =========
        Property                        Train      Dev        Test 
        ==============================  =========  =========  =========
        Quantity                        463        12         12
        ==============================  =========  =========  =========

Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Ubuntu``
    * processor: :class:`._utils.resource_processor.UbuntuResourceProcessor`
    * source: ``amazon``, ``tsinghua``
    * usage: :class:`.dataloader.UbuntuCorpus`
    * Introduction
        Ubuntu Dialogue Corpus 2.0 is a dataset containing a mass of multi-turn dialogues. The dataset has both the multi-turn property of conversations in the Dialog State Tracking Challenge datasets, and the unstructured nature of interactions from microblog services such as Twitter. Refer to the link https://github.com/rkadlec/ubuntu-ranking-dataset-creator to get more details about raw data.
    * Statistic
        =============================  ============  ============  =======
        Property                       Train         Dev           Test
        =============================  ============  ============  =======
        Quantity                       1,000,000     19,560        18,920
        minimum length of sentences    2             2             2     
        maximum length of sentences    977           343           817   
        average length of sentences    17.98         19.40         19.61 
        std of number of sentences     16.26         17.25         17.94 
        minimum number of turns        3             3             3     
        maximum number of turns        19            19            19    
        average number of turns        4.95          4.79          4.85  
        std of number of turns         2.97          2.79          2.85  
        =============================  ============  ============  =======
    * Refenence
        R. Lowe, N. Pow, I. Serban, and J. Pineau. `The ubuntu dialogue corpus: A large dataset for research in unstructured multi-turn dialogue systems <https://arxiv.org/pdf/1506.08909.pdf>`_. In Special Interest Group on Discourse and Dialogue (SIGDIAL), 2015a.

Ubuntu_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Ubuntu_small``
    * processor: :class:`._utils.resource_processor.UbuntuResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.UbuntuCorpus`
    * Statistic
        ==============================  =========  =========  =========
        Property                        Train      Dev        Test 
        ==============================  =========  =========  =========
        Quantity                        10,001     1,957      1,893
        ==============================  =========  =========  =========
