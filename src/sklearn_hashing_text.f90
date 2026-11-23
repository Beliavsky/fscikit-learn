! sklearn 1.6.1 feature_extraction/text.py HashingVectorizer and analyzers.
! sklearn BSD-3-Clause attribution: THIRD_PARTY_NOTICES.md.
! Original Fortran adapters/storage are MIT contributions.
module sklearn_hashing_text
  use sklearn_common, only: dp,require
  use sklearn_feature_extraction, only: feature_string,feature_row,feature_hasher
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: hashing_vectorizer
  type :: hashing_vectorizer
    integer :: n_features=1048576,ngram_min=1,ngram_max=1,token_min_length=2
    logical :: lowercase=.true.,alternate_sign=.true.,binary=.false.,english_stop_words=.false.
    character(8) :: analyzer='word',norm='l2'
    character(:), allocatable :: stop_words(:)
  contains
    procedure, private :: fit_chars,fit_strings
    generic :: fit=>fit_chars,fit_strings
    generic :: partial_fit=>fit_chars,fit_strings
    procedure, private :: transform_chars,transform_strings
    generic :: transform=>transform_chars,transform_strings
    procedure, private :: fit_transform_chars,fit_transform_strings
    generic :: fit_transform=>fit_transform_chars,fit_transform_strings
    procedure, private :: dense_chars,dense_strings
    generic :: transform_dense=>dense_chars,dense_strings
    generic :: fit_transform_dense=>dense_chars,dense_strings
    procedure :: analyze=>analyze_text
  end type
  ! sklearn 1.6.1 _stop_words.py ENGLISH_STOP_WORDS (BSD-3-Clause).
  character(12), parameter :: english_words(318)=[character(12):: &
    'a','about','above','across','after','afterwards', &
    'again','against','all','almost','alone','along', &
    'already','also','although','always','am','among', &
    'amongst','amoungst','amount','an','and','another', &
    'any','anyhow','anyone','anything','anyway','anywhere', &
    'are','around','as','at','back','be', &
    'became','because','become','becomes','becoming','been', &
    'before','beforehand','behind','being','below','beside', &
    'besides','between','beyond','bill','both','bottom', &
    'but','by','call','can','cannot','cant', &
    'co','con','could','couldnt','cry','de', &
    'describe','detail','do','done','down','due', &
    'during','each','eg','eight','either','eleven', &
    'else','elsewhere','empty','enough','etc','even', &
    'ever','every','everyone','everything','everywhere','except', &
    'few','fifteen','fifty','fill','find','fire', &
    'first','five','for','former','formerly','forty', &
    'found','four','from','front','full','further', &
    'get','give','go','had','has','hasnt', &
    'have','he','hence','her','here','hereafter', &
    'hereby','herein','hereupon','hers','herself','him', &
    'himself','his','how','however','hundred','i', &
    'ie','if','in','inc','indeed','interest', &
    'into','is','it','its','itself','keep', &
    'last','latter','latterly','least','less','ltd', &
    'made','many','may','me','meanwhile','might', &
    'mill','mine','more','moreover','most','mostly', &
    'move','much','must','my','myself','name', &
    'namely','neither','never','nevertheless','next','nine', &
    'no','nobody','none','noone','nor','not', &
    'nothing','now','nowhere','of','off','often', &
    'on','once','one','only','onto','or', &
    'other','others','otherwise','our','ours','ourselves', &
    'out','over','own','part','per','perhaps', &
    'please','put','rather','re','same','see', &
    'seem','seemed','seeming','seems','serious','several', &
    'she','should','show','side','since','sincere', &
    'six','sixty','so','some','somehow','someone', &
    'something','sometime','sometimes','somewhere','still','such', &
    'system','take','ten','than','that','the', &
    'their','them','themselves','then','thence','there', &
    'thereafter','thereby','therefore','therein','thereupon','these', &
    'they','thick','thin','third','this','those', &
    'though','three','through','throughout','thru','thus', &
    'to','together','too','top','toward','towards', &
    'twelve','twenty','two','un','under','until', &
    'up','upon','us','very','via','was', &
    'we','well','were','what','whatever','when', &
    'whence','whenever','where','whereafter','whereas','whereby', &
    'wherein','whereupon','wherever','whether','which','while', &
    'whither','who','whoever','whole','whom','whose', &
    'why','will','with','within','without','would', &
    'yet','you','your','yours','yourself','yourselves']
contains
  subroutine ascii_text(text)
    character(*), intent(in) :: text
    integer :: i
    call require(len(text)<huge(1),'HashingVectorizer document exceeds integer range')
    do i=1,len(text)
      call require(iachar(text(i:i))<128,'HashingVectorizer currently requires ASCII text')
    end do
  end subroutine
  subroutine controls(self)
    class(hashing_vectorizer), intent(in) :: self
    integer :: i
    call require(self%n_features>0,'HashingVectorizer n_features must be positive')
    call require(self%ngram_min>=1.and.self%ngram_max>=self%ngram_min,'HashingVectorizer invalid ngram range')
    call require(self%token_min_length>=1,'HashingVectorizer token_min_length must be positive')
    call require(any(self%analyzer==[character(8)::'word','char','char_wb']),'HashingVectorizer unknown analyzer')
    call require(any(self%norm==[character(8)::'none','l1','l2']),'HashingVectorizer unknown norm')
    call require(.not.(self%english_stop_words.and.allocated(self%stop_words)), &
      'HashingVectorizer choose English or custom stop words')
    if(allocated(self%stop_words)) then
      do i=1,size(self%stop_words)
        call ascii_text(self%stop_words(i))
      end do
    end if
  end subroutine
  subroutine fit_chars(self,documents)
    class(hashing_vectorizer), intent(inout) :: self
    character(*), optional, intent(in) :: documents(:)
    call controls(self)
  end subroutine
  subroutine fit_strings(self,documents)
    class(hashing_vectorizer), intent(inout) :: self
    type(feature_string), intent(in) :: documents(:)
    call controls(self)
  end subroutine
  subroutine append(buffer,used,text)
    type(feature_string), allocatable, intent(inout) :: buffer(:)
    integer, intent(inout) :: used
    character(*), intent(in) :: text
    type(feature_string), allocatable :: grown(:)
    integer :: capacity
    if(.not.allocated(buffer)) allocate(buffer(16))
    if(used==size(buffer)) then
      call require(size(buffer)<huge(1)/2,'HashingVectorizer token count exceeds integer range')
      capacity=2*size(buffer);allocate(grown(capacity));grown(:used)=buffer(:used)
      call move_alloc(grown,buffer)
    end if
    used=used+1;buffer(used)%text=text
  end subroutine
  logical function whitespace(ch) result(yes)
    character, intent(in) :: ch
    integer :: c
    c=iachar(ch);yes=(c>=9.and.c<=13).or.(c>=28.and.c<=32)
  end function
  logical function word_character(ch) result(yes)
    character, intent(in) :: ch
    integer :: c
    c=iachar(ch)
    yes=(c>=65.and.c<=90).or.(c>=97.and.c<=122).or.(c>=48.and.c<=57).or.c==95
  end function
  logical function stopped(self,word) result(yes)
    class(hashing_vectorizer), intent(in) :: self
    character(*), intent(in) :: word
    yes=.false.
    if(allocated(self%stop_words)) yes=any(self%stop_words==word)
    if(self%english_stop_words) yes=english_word(word)
  end function
  function analyze_text(self,document) result(tokens)
    class(hashing_vectorizer), intent(in) :: self
    character(*), intent(in) :: document
    type(feature_string), allocatable :: tokens(:),buffer(:),words(:),word_buffer(:)
    character(:), allocatable :: text,normalized,gram,padded
    integer :: i,j,n,start,used,nwords,length,run,last,c
    call controls(self);call ascii_text(document)
    text=document;length=len(text)
    if(self%lowercase) then
      do i=1,length
        c=iachar(text(i:i));if(c>=65.and.c<=90) text(i:i)=achar(c+32)
      end do
    end if
    used=0;nwords=0;allocate(buffer(16),word_buffer(16))
    if(self%analyzer=='word') then
      i=1
      do while(i<=length)
        if(.not.word_character(text(i:i))) then
          i=i+1;cycle
        end if
        start=i;i=i+1
        do while(i<=length)
          if(.not.word_character(text(i:i))) exit
          i=i+1
        end do
        if(i-start<self%token_min_length) cycle
        if(stopped(self,text(start:i-1))) cycle
        call append(word_buffer,nwords,text(start:i-1))
      end do
      words=word_buffer(:nwords)
      do n=self%ngram_min,min(self%ngram_max,nwords)
        do i=1,nwords-n+1
          gram=words(i)%text
          do j=i+1,i+n-1
            gram=gram//' '//words(j)%text
          end do
          call append(buffer,used,gram)
        end do
      end do
    else
      ! Python re.sub(r'\s\s+', ' ', text) preserves a SINGLE whitespace byte.
      allocate(character(length)::normalized);i=1;last=0
      do while(i<=length)
        if(whitespace(text(i:i))) then
          start=i;i=i+1
          do while(i<=length)
            if(.not.whitespace(text(i:i))) exit
            i=i+1
          end do
          run=i-start;last=last+1
          normalized(last:last)=text(start:start)
          if(run>=2) normalized(last:last)=' '
        else
          last=last+1;normalized(last:last)=text(i:i);i=i+1
        end if
      end do
      text=normalized(:last);length=last
      if(self%analyzer=='char') then
        do n=self%ngram_min,min(self%ngram_max,length)
          do i=1,length-n+1
            call append(buffer,used,text(i:i+n-1))
          end do
        end do
      else
        i=1
        do while(i<=length)
          if(whitespace(text(i:i))) then
            i=i+1;cycle
          end if
          start=i;i=i+1
          do while(i<=length)
            if(whitespace(text(i:i))) exit
            i=i+1
          end do
          call require(i-start<huge(1)-2,'HashingVectorizer padded word exceeds integer range')
          padded=' '//text(start:i-1)//' '
          n=self%ngram_min
          do
            do j=1,max(1,len(padded)-n+1)
              last=min(n,len(padded)-j+1)
              call append(buffer,used,padded(j:j+last-1))
            end do
            if(n>=len(padded).or.n==self%ngram_max) exit
            n=n+1
          end do
        end do
      end if
    end if
    tokens=buffer(:used)
  end function
  function transform_strings(self,documents) result(matrix)
    class(hashing_vectorizer), intent(in) :: self
    type(feature_string), intent(in) :: documents(:)
    type(csr_matrix) :: matrix
    type(feature_row), allocatable :: rows(:)
    type(feature_hasher) :: hasher
    integer :: i,left,right
    real(dp) :: scale
    call controls(self)
    call require(size(documents,kind=int64)>0.and.size(documents,kind=int64)<huge(1), &
      'HashingVectorizer document sequence must be nonempty')
    allocate(rows(size(documents)))
    do i=1,size(documents)
      call require(allocated(documents(i)%text),'HashingVectorizer document is unallocated')
      rows(i)%tokens=self%analyze(documents(i)%text)
    end do
    hasher%n_features=self%n_features;hasher%input_type='string';hasher%alternate_sign=self%alternate_sign
    matrix=hasher%transform(rows)
    ! Binary acts on STORED entries, including exact-zero signed collisions.
    if(self%binary) matrix%data=1
    do i=1,matrix%n_rows
      left=matrix%indptr(i);right=matrix%indptr(i+1)-1
      scale=0
      if(self%norm=='l1') scale=sum(abs(matrix%data(left:right)))
      if(self%norm=='l2') scale=sqrt(sum(matrix%data(left:right)**2))
      if(scale>0) matrix%data(left:right)=matrix%data(left:right)/scale
    end do
  end function
  function transform_chars(self,documents) result(matrix)
    class(hashing_vectorizer), intent(in) :: self
    character(*), intent(in) :: documents(:)
    type(csr_matrix) :: matrix
    type(feature_string), allocatable :: strings(:)
    integer :: i
    allocate(strings(size(documents)))
    do i=1,size(documents)
      strings(i)%text=trim(documents(i))
    end do
    matrix=self%transform(strings)
  end function
  function fit_transform_strings(self,documents) result(matrix)
    class(hashing_vectorizer), intent(inout) :: self
    type(feature_string), intent(in) :: documents(:)
    type(csr_matrix) :: matrix
    call self%fit(documents);matrix=self%transform(documents)
  end function
  function fit_transform_chars(self,documents) result(matrix)
    class(hashing_vectorizer), intent(inout) :: self
    character(*), intent(in) :: documents(:)
    type(csr_matrix) :: matrix
    call self%fit(documents);matrix=self%transform(documents)
  end function
  function dense_strings(self,documents) result(values)
    class(hashing_vectorizer), intent(in) :: self
    type(feature_string), intent(in) :: documents(:)
    real(dp), allocatable :: values(:,:)
    type(csr_matrix) :: matrix
    matrix=self%transform(documents);values=matrix%to_dense()
  end function
  function dense_chars(self,documents) result(values)
    class(hashing_vectorizer), intent(in) :: self
    character(*), intent(in) :: documents(:)
    real(dp), allocatable :: values(:,:)
    type(csr_matrix) :: matrix
    matrix=self%transform(documents);values=matrix%to_dense()
  end function
  logical function english_word(word) result(yes)
    character(*), intent(in) :: word
    integer :: lo,hi,mid
    lo=1;hi=size(english_words);yes=.false.
    do while(lo<=hi)
      mid=lo+(hi-lo)/2
      if(english_words(mid)==word) then
        yes=.true.;return
      else if(english_words(mid)<word) then
        lo=mid+1
      else
        hi=mid-1
      end if
    end do
  end function
end module
