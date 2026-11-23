program test_hashing_vectorizer
  use sklearn_hashing_text, only: hashing_vectorizer
  use sklearn_feature_extraction, only: feature_string
  use sklearn_sparse, only: csr_matrix
  use sklearn_common, only: dp,require
  implicit none
  type(hashing_vectorizer) :: model
  type(csr_matrix) :: matrix,other
  type(feature_string), allocatable :: tokens(:)
  type(feature_string) :: docs(1)
  real(dp), allocatable :: dense(:,:)
  matrix=model%transform([''])
  call require(matrix%n_cols==1048576.and.matrix%n_rows==1.and.matrix%nnz()==0,'sparse default width and empty text')
  model%n_features=1;model%norm='none'
  matrix=model%transform(['foo bar'])
  call require(matrix%nnz()==1.and.matrix%data(1)==0,'signed cancellation retains stored zero')
  model%binary=.true.;matrix=model%transform(['foo bar'])
  call require(matrix%data(1)==1,'binary acts after collision, including stored zero')
  model%binary=.false.;model%norm='l2';matrix=model%transform(['foo foo bar'])
  call require(matrix%data(1)==-1,'signed normalization')
  model=hashing_vectorizer();model%ngram_max=2
  model%stop_words=[character(3)::'the','and']
  tokens=model%analyze(' The red and BLUE ')
  call require(size(tokens)==3,'stop words removed before ngrams')
  call require(tokens(1)%text=='red'.and.tokens(2)%text=='blue'.and.tokens(3)%text=='red blue','word ngram order')
  model=hashing_vectorizer();model%english_stop_words=.true.
  tokens=model%analyze('this is a useful document nevertheless')
  call require(size(tokens)==2,'English stop words')
  call require(tokens(1)%text=='useful'.and.tokens(2)%text=='document','English table lookup')
  model=hashing_vectorizer();model%analyzer='char_wb';model%ngram_min=4;model%ngram_max=6
  tokens=model%analyze('a')
  call require(size(tokens)==1.and.len(tokens(1)%text)==3,'short padded word emitted once')
  call require(tokens(1)%text==' a ','short padded word boundaries')
  model%ngram_min=huge(1);model%ngram_max=huge(1)
  tokens=model%analyze('a')
  call require(size(tokens)==1.and.len(tokens(1)%text)==3,'maximum integer ngram does not overflow a DO loop')
  model%analyzer='char';model%ngram_min=1;model%ngram_max=1
  tokens=model%analyze('a'//achar(9)//'b'//achar(9)//achar(10)//'c')
  call require(size(tokens)==5.and.tokens(2)%text==achar(9).and.tokens(4)%text==' ', &
    'only whitespace runs of length two or more are collapsed')
  docs(1)%text=' x ';tokens=model%analyze(docs(1)%text)
  call require(size(tokens)==3,'exact dynamic document preserves trailing space')
  model%n_features=17;model%norm='l1'
  matrix=model%transform(docs);other=model%fit_transform(docs)
  call require(all(matrix%data==other%data).and.all(matrix%indices==other%indices),'stateless transform before fit')
  call model%fit();call model%partial_fit(docs)
  dense=model%transform_dense(docs)
  call require(maxval(abs(dense-matrix%to_dense()))==0,'dense helper agrees with sparse output')
  call require(abs(sum(abs(matrix%data))-1)<1.e-15_dp,'L1 normalized row')
  model=hashing_vectorizer();model%token_min_length=1
  tokens=model%analyze('x _ 7 __ hello')
  call require(size(tokens)==5,'single-character word pattern encoding')
  tokens=model%analyze(repeat('a',1024))
  call require(size(tokens)==1.and.len(tokens(1)%text)==1024,'unbounded dynamic token length')
  print *, 'HashingVectorizer: PASS'
end program
