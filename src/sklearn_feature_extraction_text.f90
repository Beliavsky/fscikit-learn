module sklearn_feature_extraction_text
  use sklearn_hashing_text, only: hashing_vectorizer
  use sklearn_common
  use sklearn_preprocessing, only: normalize
  implicit none
  private
  public :: count_vectorizer,tfidf_transformer,tfidf_vectorizer,hashing_vectorizer
  integer, parameter :: word_length=128
  type :: count_vectorizer
    logical :: lowercase=.true.,binary=.false.
    integer :: min_df=1,max_features=0
    real(dp) :: max_df=1
    character(word_length), allocatable :: vocabulary_(:)
  contains
    procedure :: fit=>count_fit
    procedure :: transform=>count_transform
    procedure :: fit_transform=>count_fit_transform
    procedure :: get_feature_names_out=>feature_names
  end type
  type :: tfidf_transformer
    logical :: use_idf=.true.,smooth_idf=.true.,sublinear_tf=.false.
    character(8) :: norm='l2'
    real(dp), allocatable :: idf_(:)
  contains
    procedure :: fit=>tfidf_fit
    procedure :: transform=>tfidf_transform
    procedure :: fit_transform=>tfidf_fit_transform
  end type
  type, extends(count_vectorizer) :: tfidf_vectorizer
    type(tfidf_transformer) :: tfidf
  contains
    procedure :: fit=>tfidf_vectorizer_fit
    procedure :: transform=>tfidf_vectorizer_transform
  end type
contains
  function tokenize(document,lowercase) result(words)
    character(*), intent(in) :: document
    logical, intent(in) :: lowercase
    character(word_length), allocatable :: words(:),buffer(:)
    character :: ch
    character(:), allocatable :: text
    integer :: i,start,k,code,n
    logical :: valid
    n=len_trim(document); text=document; allocate(buffer(n/2+1)); k=0; start=0
    do i=1,n+1
      valid=.false.
      if(i<=n) then
        code=iachar(text(i:i))
        if(lowercase.and.code>=65.and.code<=90) text(i:i)=achar(code+32)
        code=iachar(text(i:i))
        valid=(code>=65.and.code<=90).or.(code>=97.and.code<=122).or. &
          (code>=48.and.code<=57).or.code==95
      end if
      if(valid) then
        if(start==0) start=i
      else if(start>0) then
        if(i-start>=2) then
          call require(i-start<=word_length,'text token exceeds 128 characters')
          k=k+1; buffer(k)=text(start:i-1)
        end if
        start=0
      end if
    end do
    words=buffer(:k)
  end function
  integer function word_index(vocabulary,word) result(idx)
    character(*), intent(in) :: vocabulary(:),word
    integer :: lo,hi,middle
    lo=1; hi=size(vocabulary); idx=0
    do while(lo<=hi)
      middle=(lo+hi)/2
      if(vocabulary(middle)==word) then
        idx=middle; return
      else if(vocabulary(middle)<word) then
        lo=middle+1
      else
        hi=middle-1
      end if
    end do
  end function
  subroutine count_fit(self,documents)
    class(count_vectorizer), intent(inout) :: self
    character(*), intent(in) :: documents(:)
    character(word_length), allocatable :: all_words(:),words(:),sorted(:),vocabulary(:)
    character(word_length) :: key
    real(dp), allocatable :: counts(:,:),totals(:)
    integer, allocatable :: order(:),df(:)
    logical, allocatable :: keep(:),selected(:)
    integer :: i,j,k,n
    call require(size(documents)>0,'text documents must be nonempty')
    call require(self%min_df>=1.and.self%max_df>0.and.self%max_df<=1.and.self%max_features>=0, &
      'invalid vectorizer controls')
    allocate(all_words(0))
    do i=1,size(documents)
      words=tokenize(documents(i),self%lowercase); all_words=[all_words,words]
    end do
    call require(size(all_words)>0,'empty text vocabulary')
    do i=2,size(all_words)
      key=all_words(i); j=i-1
      do while(j>=1)
        if(all_words(j)<=key) exit
        all_words(j+1)=all_words(j); j=j-1
      end do
      all_words(j+1)=key
    end do
    k=1
    do i=2,size(all_words)
      if(all_words(i)/=all_words(k)) then
        k=k+1; all_words(k)=all_words(i)
      end if
    end do
    self%vocabulary_=all_words(:k); counts=count_transform(self,documents)
    df=count(counts>0,dim=1); totals=sum(counts,dim=1)
    keep=df>=self%min_df.and.real(df,dp)<=self%max_df*size(documents)
    if(self%max_features>0.and.count(keep)>self%max_features) then
      where(.not.keep) totals=-1
      order=argsort(-totals); allocate(selected(k)); selected=.false.
      selected(order(:self%max_features))=.true.; keep=keep.and.selected
    end if
    call require(any(keep),'no terms remain after document frequency filtering')
    self%vocabulary_=pack(self%vocabulary_,keep)
  end subroutine
  function count_transform(self,documents) result(counts)
    class(count_vectorizer), intent(in) :: self
    character(*), intent(in) :: documents(:)
    real(dp), allocatable :: counts(:,:)
    character(word_length), allocatable :: words(:)
    integer :: i,j,k
    call require(allocated(self%vocabulary_),'count vectorizer is not fitted')
    allocate(counts(size(documents),size(self%vocabulary_))); counts=0
    do i=1,size(documents)
      words=tokenize(documents(i),self%lowercase)
      do j=1,size(words)
        k=word_index(self%vocabulary_,words(j))
        if(k>0) counts(i,k)=counts(i,k)+1
      end do
    end do
    if(self%binary) counts=min(1._dp,counts)
  end function
  function count_fit_transform(self,documents) result(counts)
    class(count_vectorizer), intent(inout) :: self
    character(*), intent(in) :: documents(:)
    real(dp), allocatable :: counts(:,:)
    call self%fit(documents); counts=self%transform(documents)
  end function
  function feature_names(self) result(names)
    class(count_vectorizer), intent(in) :: self
    character(word_length), allocatable :: names(:)
    call require(allocated(self%vocabulary_),'count vectorizer is not fitted'); names=self%vocabulary_
  end function
  subroutine tfidf_fit(self,x)
    class(tfidf_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: df(size(x,2)),n
    call check_x(x); call require(all(x>=0),'TF-IDF requires nonnegative counts')
    df=real(count(x>0,dim=1),dp); n=size(x,1); self%idf_=df*0+1
    if(self%smooth_idf) then
      df=df+1; n=n+1
    end if
    if(self%use_idf) then
      call require(all(df>0),'unsmoothed IDF has an unseen feature')
      self%idf_=log(n/df)+1
    end if
  end subroutine
  function tfidf_transform(self,x) result(z)
    class(tfidf_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%idf_),'TF-IDF transformer is not fitted')
    call check_xy(transpose(x),size(self%idf_)); call require(all(x>=0),'TF-IDF requires nonnegative counts')
    z=x
    if(self%sublinear_tf) then
      where(x>0) z=1+log(max(x,tiny(1._dp)))
    end if
    if(self%use_idf) z=z*spread(self%idf_,1,size(x,1))
    if(trim(self%norm)/='none') z=normalize(z,self%norm)
  end function
  function tfidf_fit_transform(self,x) result(z)
    class(tfidf_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  subroutine tfidf_vectorizer_fit(self,documents)
    class(tfidf_vectorizer), intent(inout) :: self
    character(*), intent(in) :: documents(:)
    real(dp), allocatable :: counts(:,:)
    call count_fit(self,documents); counts=count_transform(self,documents); call self%tfidf%fit(counts)
  end subroutine
  function tfidf_vectorizer_transform(self,documents) result(z)
    class(tfidf_vectorizer), intent(in) :: self
    character(*), intent(in) :: documents(:)
    real(dp), allocatable :: z(:,:),counts(:,:)
    counts=count_transform(self,documents); z=self%tfidf%transform(counts)
  end function
end module
