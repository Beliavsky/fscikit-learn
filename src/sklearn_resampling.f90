! Dense splitters adapted from sklearn 1.6.1 model_selection/_split.py and
! utils/extmath.py _approximate_mode (BSD-3-Clause). See THIRD_PARTY_NOTICES.md.
module sklearn_resampling
  use sklearn_common, only: dp,require,unique_labels
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: iso_c_binding, only: c_char,c_ptr,c_double,c_null_char,c_null_ptr
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_negative
  implicit none
  private
  public :: index_split,shuffle_split,stratified_shuffle_split,group_shuffle_split
  public :: repeated_kfold,repeated_stratified_kfold,leave_p_out,leave_p_groups_out
  public :: resample,shuffle
  ! Internal numeric rendering helper shared by classification reports.
  public :: real_text
  interface resample
    module procedure resample_population,resample_iv,resample_im,resample_jv,resample_jm
    module procedure resample_rv,resample_rm,resample_bv,resample_bm,resample_cv,resample_cm,resample_sparse
  end interface
  interface shuffle
    module procedure shuffle_population,shuffle_iv,shuffle_im,shuffle_jv,shuffle_jm
    module procedure shuffle_rv,shuffle_rm,shuffle_bv,shuffle_bm,shuffle_cv,shuffle_cm,shuffle_sparse
  end interface
  interface
    function c_strtod(text,tail) bind(C,name='strtod') result(value)
      import c_char,c_ptr,c_double
      character(c_char), intent(in) :: text(*)
      type(c_ptr), value :: tail
      real(c_double) :: value
    end function
  end interface
  type :: index_split
    integer, allocatable :: train(:),test(:)
  end type
  type :: shuffle_controls
    integer :: random_state=0,test_count=0,train_count=0
    real(dp) :: test_size=-1,train_size=-1
  contains
    procedure :: get_n_splits=>shuffle_count
  end type
  type, extends(shuffle_controls) :: shuffle_split
    integer :: n_splits=10
  contains
    procedure :: split=>random_split
  end type
  type, extends(shuffle_controls) :: stratified_shuffle_split
    integer :: n_splits=10
  contains
    procedure :: split=>stratified_random_split
  end type
  type, extends(shuffle_controls) :: group_shuffle_split
    integer :: n_splits=5
  contains
    procedure :: split=>group_random_split
  end type
  type :: repeat_controls
    integer :: n_splits=5,n_repeats=10,random_state=0
  contains
    procedure :: get_n_splits=>repeat_count
  end type
  type, extends(repeat_controls) :: repeated_kfold
  contains
    procedure :: split=>repeated_split
  end type
  type, extends(repeat_controls) :: repeated_stratified_kfold
  contains
    procedure :: split=>repeated_stratified_split
  end type
  type :: leave_p_out
    integer :: p=2
  contains
    procedure :: split=>leave_p_split
    procedure :: get_n_splits=>leave_p_count
  end type
  type :: leave_p_groups_out
    integer :: n_groups=2
  contains
    procedure :: split=>leave_groups_split
    procedure :: get_n_splits=>leave_groups_count
  end type
contains
  function integer_text(value) result(text)
    integer(int64), intent(in) :: value
    character(:), allocatable :: text
    character(32) :: buffer
    write(buffer,'(I0)') value;text=trim(buffer)
  end function
  function real_text(value) result(text)
    ! Shortest round-trip decimal, using the NumPy float64 scalar thresholds.
    real(dp), intent(in) :: value
    character(:), allocatable :: text,digits,mantissa,exponent_text
    character(64) :: buffer,format_
    integer :: precision,exponent_,position,i,e,scale_,delta
    integer(int64) :: coefficient,neighbor
    logical :: negative,found
    negative=ieee_is_negative(value)
    if(value==0) then
      text='0.0';if(negative) text='-0.0';return
    end if
    do precision=1,17
      write(format_,'(A,I0,A)') '(ES26.',precision-1,'E3)'
      write(buffer,format_) abs(value)
      buffer=adjustl(buffer);e=index(buffer,'E');read(buffer(e+1:),*) exponent_
      digits=''
      do i=1,e-1
        if(buffer(i:i)/='.'.and.buffer(i:i)/=' ') digits=digits//buffer(i:i)
      end do
      if(decimal_roundtrips(digits,exponent_,abs(value))) exit
      ! At powers of two the rounding interval is asymmetric. A neighboring
      ! decimal can be shortest even when the nearest decimal misses it.
      read(digits,*) coefficient;scale_=exponent_-len(digits)+1;found=.false.
      do delta=-1,1,2
        neighbor=coefficient+delta
        if(neighbor<=0) cycle
        mantissa=integer_text(neighbor)
        position=scale_+len(mantissa)-1
        if(decimal_roundtrips(mantissa,position,abs(value))) then
          digits=mantissa;exponent_=position;found=.true.;exit
        end if
      end do
      if(found) exit
    end do
    do while(len(digits)>1)
      if(digits(len(digits):)/='0') exit
      digits=digits(:len(digits)-1)
    end do
    if(exponent_<-4.or.exponent_>=16) then
      mantissa=digits(:1)
      if(len(digits)>1) mantissa=mantissa//'.'//digits(2:)
      exponent_text=integer_text(int(abs(exponent_),int64))
      if(len(exponent_text)<2) exponent_text='0'//exponent_text
      if(exponent_<0) then
        text=mantissa//'e-'//exponent_text
      else
        text=mantissa//'e+'//exponent_text
      end if
    else
      position=exponent_+1
      if(position<=0) then
        text='0.'//repeat('0',-position)//digits
      else if(position>=len(digits)) then
        text=digits//repeat('0',position-len(digits))//'.0'
      else
        text=digits(:position)//'.'//digits(position+1:)
      end if
    end if
    if(negative) text='-'//text
  end function
  logical function decimal_roundtrips(digits,exponent_,value) result(equal)
    character(*), intent(in) :: digits
    integer, intent(in) :: exponent_
    real(dp), intent(in) :: value
    character(:), allocatable :: clean,text
    real(dp) :: parsed
    clean=digits;equal=.false.
    do while(len(clean)>1)
      if(clean(len(clean):)/='0') exit
      clean=clean(:len(clean)-1)
    end do
    if(exponent_==308) then
      ! Coarse decimal rounding of HUGE can otherwise overflow the reader.
      if(clean//repeat('0',17-len(clean))>'17976931348623157') return
    end if
    text=clean(:1)//'.'//clean(2:)//'E'//integer_text(int(exponent_,int64))
    ! Direct double parsing avoids double rounding through an extended
    ! intermediate in some Fortran formatted-input implementations.
    parsed=c_strtod(text//c_null_char,c_null_ptr);equal=parsed==value
  end function
  function stratification_ids(stratify) result(labels)
    class(*), intent(in) :: stratify(..)
    integer, allocatable :: labels(:),order(:),work(:)
    integer(int64), allocatable :: integers(:)
    real(dp), allocatable :: reals(:)
    character(:), allocatable :: keys(:),word
    integer :: kind_,n,q,width,i,j,position,left,middle,right,a,b,k,run,count_
    integer(int64) :: capacity
    select rank(stratify)
    rank(1)
      n=size(stratify);call require(n>0,'resample stratification must be nonempty')
      select type(stratify)
      type is(integer);kind_=1;integers=int(stratify,int64)
      type is(integer(int64));kind_=1;integers=stratify
      type is(logical);kind_=1;integers=merge(1_int64,0_int64,stratify)
      type is(real(dp))
        call require(all(ieee_is_finite(stratify)),'resample stratification must be finite')
        kind_=2;reals=stratify
      type is(character(*));kind_=3;keys=stratify
      class default;call require(.false.,'resample stratification type is unsupported')
      end select
    rank(2)
      n=size(stratify,1);q=size(stratify,2)
      call require(n>0.and.q>0,'resample stratification needs nonempty axes')
      width=26
      select type(stratify)
      type is(character(*));width=max(1,len(stratify))+1
      class default
      end select
      capacity=int(width,int64)*q
      call require(capacity<=huge(1),'resample stratification row text exceeds integer range')
      allocate(character(len=int(capacity))::keys(n));keys='';kind_=3
      do i=1,n
        position=1
        do j=1,q
          select type(stratify)
          type is(integer);word=integer_text(int(stratify(i,j),int64))
          type is(integer(int64));word=integer_text(stratify(i,j))
          type is(logical)
            word='False';if(stratify(i,j)) word='True'
          type is(real(dp))
            call require(ieee_is_finite(stratify(i,j)),'resample stratification must be finite')
            word=real_text(stratify(i,j))
          type is(character(*));word=stratify(i,j)
          class default;call require(.false.,'resample stratification type is unsupported')
          end select
          keys(i)(position:position+len(word)-1)=word;position=position+len(word)
          if(j<q) then
            keys(i)(position:position)=' ';position=position+1
          end if
        end do
      end do
    rank default;call require(.false.,'resample stratification must be a vector or matrix')
    end select
    allocate(order(n),work(n),labels(n));order=[(i,i=1,n)];run=1
    do while(run<n)
      left=1
      do while(left<=n)
        middle=left+min(run,n-left+1)-1;right=middle+min(run,n-middle);a=left;b=middle+1
        do k=left,right
          if(a>middle) then
            work(k)=order(b);b=b+1
          else if(b>right) then
            work(k)=order(a);a=a+1
          else if(before(order(b),order(a))) then
            work(k)=order(b);b=b+1
          else
            work(k)=order(a);a=a+1
          end if
        end do
        left=right+1
      end do
      order=work;if(run>=n-run) exit
      run=run*2
    end do
    count_=1;labels(order(1))=count_
    do i=2,n
      if(before(order(i-1),order(i))) count_=count_+1
      labels(order(i))=count_
    end do
  contains
    logical function before(a,b) result(value)
      integer, intent(in) :: a,b
      select case(kind_)
      case(1);value=integers(a)<integers(b)
      case(2);value=reals(a)<reals(b)
      case(3);value=keys(a)<keys(b)
      end select
    end function
  end function
  function resample_population(n_population,n_samples,replace,random_state,stratify) result(indices)
    integer, intent(in) :: n_population
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    integer, allocatable :: indices(:),labels(:),counts(:),allocations(:),members(:),order(:),ptr(:),sorted(:),next(:)
    integer :: m,seed,c,i,j,position,nclasses,n
    logical :: replacement
    type(legacy_random_state) :: rng
    call require(n_population>=0,'resample population size must be nonnegative')
    m=n_population
    if(present(n_samples)) then
      call require(n_samples>=1,'resample explicit n_samples must be positive');m=n_samples
    end if
    replacement=.true.;if(present(replace)) replacement=replace
    if(.not.replacement) call require(m<=n_population,'resample cannot draw more rows without replacement')
    seed=0;if(present(random_state)) seed=random_state
    call rng%seed(seed)
    if(present(stratify)) then
      labels=stratification_ids(stratify);nclasses=maxval(labels);n=size(labels)
      call require(m>0,'resample baseline rejects empty stratified index arrays')
      allocate(counts(nclasses));counts=0
      do i=1,n
        counts(labels(i))=counts(labels(i))+1
      end do
      allocations=approximate_mode(counts,m,rng)
      allocate(indices(sum(allocations)),ptr(nclasses+1),next(nclasses),sorted(n));ptr(1)=1
      do c=1,nclasses
        ptr(c+1)=ptr(c)+counts(c)
      end do
      next=ptr(:nclasses)
      do i=1,n
        c=labels(i);sorted(next(c))=i;next(c)=next(c)+1
      end do
      position=1
      do c=1,nclasses
        members=sorted(ptr(c):ptr(c+1)-1)
        if(replacement) then
          do j=1,allocations(c)
            indices(position)=members(rng%integer_index(size(members)));position=position+1
          end do
        else
          call require(allocations(c)<=counts(c),'resample stratified class draw exceeds its population')
          order=permutation(size(members),rng)
          indices(position:position+allocations(c)-1)=members(order(:allocations(c)));position=position+allocations(c)
        end if
      end do
      order=permutation(size(indices),rng);indices=indices(order)
      call require(all(indices<=n_population),'resample stratification selected rows beyond the data')
    else
      allocate(indices(m))
      if(replacement) then
        if(m>0) call require(n_population>0,'resample cannot draw from an empty population')
        do i=1,m
          indices(i)=rng%integer_index(n_population)
        end do
      else
        order=permutation(n_population,rng);indices=order(:m)
      end if
    end if
  end function
  function shuffle_population(n_population,n_samples,random_state) result(indices)
    integer, intent(in) :: n_population
    integer, optional, intent(in) :: n_samples,random_state
    integer, allocatable :: indices(:)
    indices=resample_population(n_population,n_samples,.false.,random_state)
  end function
  function permutation(n,rng) result(order)
    integer, intent(in) :: n
    type(legacy_random_state), intent(inout) :: rng
    integer :: order(n),i,j,temp
    order=[(i,i=1,n)]
    do i=n,2,-1
      j=rng%integer_index(i); temp=order(i); order(i)=order(j); order(j)=temp
    end do
  end function
  integer function shuffle_count(self) result(n)
    class(shuffle_controls), intent(in) :: self
    n=0
    select type(self)
    class is(shuffle_split); n=self%n_splits
    class is(stratified_shuffle_split); n=self%n_splits
    class is(group_shuffle_split); n=self%n_splits
    end select
    call require(n>0,'resampling n_splits must be positive')
  end function
  subroutine split_sizes(self,n,ntrain,ntest)
    class(shuffle_controls), intent(in) :: self
    integer, intent(in) :: n
    integer, intent(out) :: ntrain,ntest
    real(dp) :: tr,te,default_test
    logical :: has_train,has_test
    call require(n>=2,'shuffle split requires at least two samples or groups')
    call require(self%test_count>=0.and.self%train_count>=0,'shuffle split counts must be nonnegative')
    tr=self%train_size; te=self%test_size
    call require(ieee_is_finite(tr).and.ieee_is_finite(te),'shuffle split fractions must be finite')
    call require(tr==-1.or.(tr>0.and.tr<1),'invalid shuffle split train_size')
    call require(te==-1.or.(te>0.and.te<1),'invalid shuffle split test_size')
    call require(self%train_count==0.or.tr==-1,'shuffle split train count/fraction conflict')
    call require(self%test_count==0.or.te==-1,'shuffle split test count/fraction conflict')
    call require(self%train_count<n.and.self%test_count<n,'shuffle split count must be smaller than population')
    has_train=tr/=-1.or.self%train_count/=0; has_test=te/=-1.or.self%test_count/=0
    if(.not.has_train.and..not.has_test) then
      default_test=.1_dp
      select type(self)
      class is(group_shuffle_split); default_test=.2_dp
      end select
      te=default_test; has_test=.true.
    end if
    if(tr/=-1.and.te/=-1) call require(tr+te<=1,'shuffle split fractions sum exceeds one')
    ntrain=self%train_count; ntest=self%test_count
    if(tr/=-1) ntrain=floor(tr*n)
    if(te/=-1) ntest=ceiling(te*n)
    if(.not.has_train) ntrain=n-ntest
    if(.not.has_test) ntest=n-ntrain
    call require(ntrain<=n-ntest,'shuffle split sizes exceed population')
    call require(ntrain>0,'shuffle split training set is empty')
  end subroutine
  function random_split(self,n_samples) result(folds)
    class(shuffle_split), intent(in) :: self
    integer, intent(in) :: n_samples
    type(index_split), allocatable :: folds(:)
    type(legacy_random_state) :: rng
    integer :: ntrain,ntest,i,n
    integer, allocatable :: order(:)
    n=self%get_n_splits(); call split_sizes(self,n_samples,ntrain,ntest)
    call rng%seed(self%random_state); allocate(folds(n))
    do i=1,n
      order=permutation(n_samples,rng)
      folds(i)%test=order(:ntest); folds(i)%train=order(ntest+1:ntest+ntrain)
    end do
  end function
  function approximate_mode(counts,draws,rng) result(allocations)
    integer, intent(in) :: counts(:),draws
    type(legacy_random_state), intent(inout) :: rng
    integer :: allocations(size(counts)),need,add_now
    integer, allocatable :: ids(:),order(:)
    real(dp) :: continuous(size(counts)),remainder(size(counts)),value
    integer :: i
    continuous=real(counts,dp)/real(sum(counts),dp)*draws
    allocations=floor(continuous); need=draws-sum(allocations)
    remainder=continuous-allocations
    do while(need>0)
      value=maxval(remainder); ids=pack([(i,i=1,size(counts))],remainder==value)
      add_now=min(size(ids),need)
      ! RandomState.choice(..., replace=False) permutes even when every tie is used.
      order=permutation(size(ids),rng); ids=ids(order)
      allocations(ids(:add_now))=allocations(ids(:add_now))+1
      need=need-add_now; where(remainder==value) remainder=-1
    end do
  end function
  function stratified_random_split(self,n_samples) result(folds)
    class(stratified_shuffle_split), intent(in) :: self
    integer, intent(in) :: n_samples(:)
    type(index_split), allocatable :: folds(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: labels(:),counts(:),a(:),b(:),members(:),order(:),train(:),test(:)
    integer :: n,ntrain,ntest,c,i,j,tpos,vpos,k
    n=self%get_n_splits(); call split_sizes(self,size(n_samples),ntrain,ntest)
    labels=unique_labels(n_samples); k=size(labels); allocate(counts(k),train(ntrain),test(ntest))
    do c=1,k
      counts(c)=count(n_samples==labels(c))
    end do
    call require(all(counts>=2),'stratified shuffle classes require at least two samples')
    call require(ntrain>=k.and.ntest>=k,'stratified shuffle sizes must cover every class')
    call rng%seed(self%random_state); allocate(folds(n))
    do i=1,n
      a=approximate_mode(counts,ntrain,rng); b=approximate_mode(counts-a,ntest,rng)
      tpos=0; vpos=0
      do c=1,k
        members=pack([(j,j=1,size(n_samples))],n_samples==labels(c)); order=permutation(counts(c),rng)
        members=members(order)
        train(tpos+1:tpos+a(c))=members(:a(c)); tpos=tpos+a(c)
        test(vpos+1:vpos+b(c))=members(a(c)+1:a(c)+b(c)); vpos=vpos+b(c)
      end do
      order=permutation(ntrain,rng); folds(i)%train=train(order)
      order=permutation(ntest,rng); folds(i)%test=test(order)
    end do
  end function
  function group_random_split(self,n_samples) result(folds)
    class(group_shuffle_split), intent(in) :: self
    integer, intent(in) :: n_samples(:)
    type(index_split), allocatable :: folds(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: labels(:),order(:),ids(:)
    logical :: train(size(n_samples)),test(size(n_samples))
    integer :: ntrain,ntest,n,i,j
    labels=unique_labels(n_samples); n=self%get_n_splits(); call split_sizes(self,size(labels),ntrain,ntest)
    call rng%seed(self%random_state); allocate(folds(n)); ids=[(j,j=1,size(n_samples))]
    do i=1,n
      order=permutation(size(labels),rng); train=.false.; test=.false.
      do j=1,ntest
        test=test.or.n_samples==labels(order(j))
      end do
      do j=ntest+1,ntest+ntrain
        train=train.or.n_samples==labels(order(j))
      end do
      folds(i)%train=pack(ids,train); folds(i)%test=pack(ids,test)
    end do
  end function
  integer function repeat_count(self) result(n)
    class(repeat_controls), intent(in) :: self
    call require(self%n_splits>=2,'repeated split n_splits must be at least two')
    call require(self%n_repeats>0,'repeated split n_repeats must be positive')
    call require(self%n_repeats<=huge(n)/self%n_splits,'repeated split count exceeds native index range')
    n=self%n_splits*self%n_repeats
  end function
  function repeated_split(self,n_samples) result(folds)
    class(repeated_kfold), intent(in) :: self
    integer, intent(in) :: n_samples
    type(index_split), allocatable :: folds(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: order(:),ids(:)
    logical :: held_out(n_samples)
    integer :: total,r,k,i,start,finish,width
    total=self%get_n_splits(); call require(n_samples>=self%n_splits,'repeated split n_splits exceeds samples')
    call rng%seed(self%random_state); allocate(folds(total)); ids=[(i,i=1,n_samples)]
    do r=1,self%n_repeats
      order=permutation(n_samples,rng); start=1
      do k=1,self%n_splits
        width=n_samples/self%n_splits+merge(1,0,k<=mod(n_samples,self%n_splits)); finish=start+width-1
        held_out=.false.; held_out(order(start:finish))=.true.; i=(r-1)*self%n_splits+k
        folds(i)%train=pack(ids,.not.held_out); folds(i)%test=pack(ids,held_out); start=finish+1
      end do
    end do
  end function
  function repeated_stratified_split(self,n_samples) result(folds)
    class(repeated_stratified_kfold), intent(in) :: self
    integer, intent(in) :: n_samples(:)
    type(index_split), allocatable :: folds(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: labels(:),counts(:),ids(:),assignment(:),members(:),class_folds(:),order(:)
    integer, allocatable :: allocation(:,:)
    integer :: total,n,c,i,j,k,r,offset,position,temp
    total=self%get_n_splits(); n=self%n_splits
    call require(size(n_samples)>=n,'repeated split n_splits exceeds samples')
    labels=unique_labels(n_samples)
    ! StratifiedKFold numbers classes by first occurrence, not sorted label value.
    do i=2,size(labels)
      temp=labels(i); j=i-1
      do while(j>=1)
        if(findloc(n_samples,labels(j),dim=1)<findloc(n_samples,temp,dim=1)) exit
        labels(j+1)=labels(j); j=j-1
      end do
      labels(j+1)=temp
    end do
    allocate(counts(size(labels)),allocation(n,size(labels)),assignment(size(n_samples)))
    do c=1,size(labels)
      counts(c)=count(n_samples==labels(c))
    end do
    call require(any(counts>=n),'repeated stratified n_splits exceeds every class count')
    allocation=0; offset=0
    do c=1,size(labels)
      do i=1,counts(c)
        j=1+mod(offset+i-1,n); allocation(j,c)=allocation(j,c)+1
      end do
      offset=offset+counts(c)
    end do
    call rng%seed(self%random_state); allocate(folds(total)); ids=[(i,i=1,size(n_samples))]
    do r=1,self%n_repeats
      do c=1,size(labels)
        members=pack(ids,n_samples==labels(c)); class_folds=members; position=1
        do k=1,n
          class_folds(position:position+allocation(k,c)-1)=k; position=position+allocation(k,c)
        end do
        order=permutation(counts(c),rng); assignment(members)=class_folds(order)
      end do
      do k=1,n
        i=(r-1)*n+k; folds(i)%train=pack(ids,assignment/=k); folds(i)%test=pack(ids,assignment==k)
      end do
    end do
  end function
  integer function combinations_count(n,p) result(total)
    integer, intent(in) :: n,p
    integer(int64) :: count64
    integer :: i,k
    call require(p>=1.and.p<n,'leave-p requires 1 <= p < population')
    k=min(p,n-p); count64=1
    do i=1,k
      count64=count64*int(n-k+i,int64)/i
      call require(count64<=int(huge(total),int64),'leave-p split count exceeds native index range')
    end do
    total=int(count64)
  end function
  integer function leave_p_count(self,n_samples) result(total)
    class(leave_p_out), intent(in) :: self
    integer, intent(in) :: n_samples
    total=combinations_count(n_samples,self%p)
  end function
  integer function leave_groups_count(self,groups) result(total)
    class(leave_p_groups_out), intent(in) :: self
    integer, intent(in) :: groups(:)
    integer, allocatable :: labels(:)
    labels=unique_labels(groups); total=combinations_count(size(labels),self%n_groups)
  end function
  subroutine next_combination(chosen,n)
    integer, intent(inout) :: chosen(:)
    integer, intent(in) :: n
    integer :: i,j,p
    p=size(chosen)
    do i=p,1,-1
      if(chosen(i)<n-p+i) then
        chosen(i)=chosen(i)+1
        do j=i+1,p
          chosen(j)=chosen(j-1)+1
        end do
        return
      end if
    end do
  end subroutine
  function leave_p_split(self,n_samples) result(folds)
    class(leave_p_out), intent(in) :: self
    integer, intent(in) :: n_samples
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: chosen(:),ids(:)
    logical, allocatable :: held_out(:)
    integer :: total,i,ierr
    total=self%get_n_splits(n_samples)
    allocate(folds(total),held_out(n_samples),stat=ierr)
    call require(ierr==0,'leave-p split allocation failed')
    chosen=[(i,i=1,self%p)]; ids=[(i,i=1,n_samples)]
    do i=1,total
      held_out=.false.; held_out(chosen)=.true.; folds(i)%test=chosen
      folds(i)%train=pack(ids,.not.held_out)
      if(i<total) call next_combination(chosen,n_samples)
    end do
  end function
  function leave_groups_split(self,groups) result(folds)
    class(leave_p_groups_out), intent(in) :: self
    integer, intent(in) :: groups(:)
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: labels(:),chosen(:),ids(:)
    logical :: held_out(size(groups))
    integer :: total,i,j,ierr
    labels=unique_labels(groups); total=self%get_n_splits(groups)
    allocate(folds(total),stat=ierr); call require(ierr==0,'leave-p split allocation failed')
    chosen=[(i,i=1,self%n_groups)]; ids=[(i,i=1,size(groups))]
    do i=1,total
      held_out=.false.
      do j=1,size(chosen)
        held_out=held_out.or.groups==labels(chosen(j))
      end do
      folds(i)%test=pack(ids,held_out); folds(i)%train=pack(ids,.not.held_out)
      if(i<total) call next_combination(chosen,size(labels))
    end do
  end function
  function resample_iv(x,n_samples,replace,random_state,stratify) result(output)
    integer, intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    integer, allocatable :: output(:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids)
  end function
  function shuffle_iv(x,n_samples,random_state) result(output)
    integer, intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    integer, allocatable :: output(:)
    output=resample_iv(x,n_samples,.false.,random_state)
  end function
  function resample_im(x,n_samples,replace,random_state,stratify) result(output)
    integer, intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    integer, allocatable :: output(:,:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids,:)
  end function
  function shuffle_im(x,n_samples,random_state) result(output)
    integer, intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    integer, allocatable :: output(:,:)
    output=resample_im(x,n_samples,.false.,random_state)
  end function
  function resample_jv(x,n_samples,replace,random_state,stratify) result(output)
    integer(int64), intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    integer(int64), allocatable :: output(:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids)
  end function
  function shuffle_jv(x,n_samples,random_state) result(output)
    integer(int64), intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    integer(int64), allocatable :: output(:)
    output=resample_jv(x,n_samples,.false.,random_state)
  end function
  function resample_jm(x,n_samples,replace,random_state,stratify) result(output)
    integer(int64), intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    integer(int64), allocatable :: output(:,:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids,:)
  end function
  function shuffle_jm(x,n_samples,random_state) result(output)
    integer(int64), intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    integer(int64), allocatable :: output(:,:)
    output=resample_jm(x,n_samples,.false.,random_state)
  end function
  function resample_rv(x,n_samples,replace,random_state,stratify) result(output)
    real(dp), intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    real(dp), allocatable :: output(:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids)
  end function
  function shuffle_rv(x,n_samples,random_state) result(output)
    real(dp), intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    real(dp), allocatable :: output(:)
    output=resample_rv(x,n_samples,.false.,random_state)
  end function
  function resample_rm(x,n_samples,replace,random_state,stratify) result(output)
    real(dp), intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    real(dp), allocatable :: output(:,:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids,:)
  end function
  function shuffle_rm(x,n_samples,random_state) result(output)
    real(dp), intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    real(dp), allocatable :: output(:,:)
    output=resample_rm(x,n_samples,.false.,random_state)
  end function
  function resample_bv(x,n_samples,replace,random_state,stratify) result(output)
    logical, intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    logical, allocatable :: output(:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids)
  end function
  function shuffle_bv(x,n_samples,random_state) result(output)
    logical, intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, allocatable :: output(:)
    output=resample_bv(x,n_samples,.false.,random_state)
  end function
  function resample_bm(x,n_samples,replace,random_state,stratify) result(output)
    logical, intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    logical, allocatable :: output(:,:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids,:)
  end function
  function shuffle_bm(x,n_samples,random_state) result(output)
    logical, intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, allocatable :: output(:,:)
    output=resample_bm(x,n_samples,.false.,random_state)
  end function
  function resample_cv(x,n_samples,replace,random_state,stratify) result(output)
    character(*), intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    character(:), allocatable :: output(:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids)
  end function
  function shuffle_cv(x,n_samples,random_state) result(output)
    character(*), intent(in) :: x(:)
    integer, optional, intent(in) :: n_samples,random_state
    character(:), allocatable :: output(:)
    output=resample_cv(x,n_samples,.false.,random_state)
  end function
  function resample_cm(x,n_samples,replace,random_state,stratify) result(output)
    character(*), intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    character(:), allocatable :: output(:,:)
    integer, allocatable :: ids(:)
    ids=resample_population(size(x,1),n_samples,replace,random_state,stratify)
    output=x(ids,:)
  end function
  function shuffle_cm(x,n_samples,random_state) result(output)
    character(*), intent(in) :: x(:,:)
    integer, optional, intent(in) :: n_samples,random_state
    character(:), allocatable :: output(:,:)
    output=resample_cm(x,n_samples,.false.,random_state)
  end function
  function resample_sparse(x,n_samples,replace,random_state,stratify) result(output)
    type(csr_matrix), intent(in) :: x
    integer, optional, intent(in) :: n_samples,random_state
    logical, optional, intent(in) :: replace
    class(*), optional, intent(in) :: stratify(..)
    type(csr_matrix) :: output
    integer, allocatable :: ids(:)
    integer(int64) :: total
    integer :: i,start,finish,length_,position
    call x%validate()
    ids=resample_population(x%n_rows,n_samples,replace,random_state,stratify)
    call require(size(ids)<huge(1),'resample sparse output row count exceeds index range')
    total=0
    do i=1,size(ids)
      total=total+int(x%indptr(ids(i)+1)-x%indptr(ids(i)),int64)
    end do
    call require(total<huge(1),'resample sparse stored count exceeds index range')
    output%n_rows=size(ids);output%n_cols=x%n_cols
    allocate(output%indptr(size(ids)+1),output%indices(int(total)),output%data(int(total)))
    position=1;output%indptr(1)=1
    do i=1,size(ids)
      start=x%indptr(ids(i));finish=x%indptr(ids(i)+1)-1;length_=finish-start+1
      output%indices(position:position+length_-1)=x%indices(start:finish)
      output%data(position:position+length_-1)=x%data(start:finish)
      position=position+length_;output%indptr(i+1)=position
    end do
    call output%validate()
  end function
  function shuffle_sparse(x,n_samples,random_state) result(output)
    type(csr_matrix), intent(in) :: x
    integer, optional, intent(in) :: n_samples,random_state
    type(csr_matrix) :: output
    output=resample_sparse(x,n_samples,.false.,random_state)
  end function
end module
