program reference_hashing_vectorizer_driver
  use sklearn_hashing_text, only: hashing_vectorizer
  use sklearn_feature_extraction, only: feature_string
  use sklearn_sparse, only: csr_matrix
  use sklearn_common, only: dp,require
  implicit none
  type(hashing_vectorizer) :: model
  type(feature_string), allocatable :: docs(:),tokens(:),stops(:)
  character(:), allocatable :: char_docs(:)
  type(csr_matrix) :: matrix,copy
  real(dp), allocatable :: dense(:,:)
  integer :: n,analyzer,lower,sign,binary,english,norm,padded,nstops,i,j,maxlen
  character(8), parameter :: analyzers(3)=[character(8)::'word','char','char_wb']
  character(4), parameter :: norms(3)=[character(4)::'none','l1','l2']
  character(40) :: argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    allocate(docs(1));docs(1)%text='foo bar'
    select case(trim(argument))
    case('features');model%n_features=0
    case('min_gram');model%ngram_min=0
    case('gram_order');model%ngram_min=3;model%ngram_max=2
    case('token_length');model%token_min_length=0
    case('analyzer');model%analyzer='invalid'
    case('norm');model%norm='invalid'
    case('stop_conflict');model%english_stop_words=.true.;model%stop_words=['foo']
    case('unicode');docs(1)%text=achar(195)//achar(169)
    case('unicode_stop');model%stop_words=[achar(195)//achar(169)]
    case('unallocated');deallocate(docs(1)%text)
    case('empty');deallocate(docs);allocate(docs(0))
    case('fit_control');model%n_features=0;call model%fit();error stop 'invalid fit accepted'
    case('partial_control');model%norm='invalid';call model%partial_fit();error stop 'invalid partial fit accepted'
    case default;error stop 'unknown rejection case'
    end select
    matrix=model%transform(docs);error stop 'invalid input accepted'
  end if
  read(*,*) n,model%n_features,analyzer,model%ngram_min,model%ngram_max,model%token_min_length, &
    lower,sign,binary,english,norm,padded,nstops
  model%analyzer=analyzers(analyzer+1);model%norm=norms(norm+1)
  model%lowercase=lower/=0;model%alternate_sign=sign/=0;model%binary=binary/=0
  model%english_stop_words=english/=0
  allocate(stops(nstops),docs(n))
  maxlen=0
  do i=1,nstops
    call read_text(stops(i));maxlen=max(maxlen,len(stops(i)%text))
  end do
  if(nstops>0) then
    allocate(character(maxlen)::model%stop_words(nstops))
    do i=1,nstops
      model%stop_words(i)=stops(i)%text
    end do
  end if
  maxlen=0
  do i=1,n
    call read_text(docs(i));maxlen=max(maxlen,len(docs(i)%text))
  end do
  if(padded/=0) then
    allocate(character(maxlen)::char_docs(n))
    do i=1,n
      char_docs(i)=docs(i)%text
      docs(i)%text=trim(docs(i)%text)
    end do
    matrix=model%transform(char_docs);copy=model%fit_transform(char_docs)
    call model%partial_fit(char_docs)
    if(model%n_features<=4096) dense=model%fit_transform_dense(char_docs)
  else
    matrix=model%transform(docs);copy=model%fit_transform(docs)
    call model%partial_fit(docs)
    if(model%n_features<=4096) dense=model%transform_dense(docs)
  end if
  call require(all(matrix%indptr==copy%indptr).and.all(matrix%indices==copy%indices).and. &
    all(matrix%data==copy%data),'stateless calls disagree')
  if(allocated(dense)) call require(all(dense==matrix%to_dense()),'dense helper disagrees')
  call matrix%validate()
  print *,matrix%n_rows,matrix%n_cols,matrix%nnz()
  print *,matrix%indptr
  print *,matrix%indices
  print '(*(es25.16e3,1x))',matrix%data
  print *,n
  do i=1,n
    tokens=model%analyze(docs(i)%text);print *,size(tokens)
    do j=1,size(tokens)
      call write_text(tokens(j)%text)
    end do
  end do
contains
  subroutine read_text(string)
    type(feature_string), intent(out) :: string
    integer, allocatable :: codes(:)
    integer :: k,length
    read(*,*) length
    allocate(codes(length));allocate(character(length)::string%text)
    if(length>0) read(*,*) codes
    do k=1,length
      string%text(k:k)=achar(codes(k))
    end do
  end subroutine
  subroutine write_text(text)
    character(*), intent(in) :: text
    integer :: k
    print *,len(text)
    if(len(text)>0) print *,[(iachar(text(k:k)),k=1,len(text))]
  end subroutine
end program
