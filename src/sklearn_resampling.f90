! Dense splitters adapted from sklearn 1.6.1 model_selection/_split.py and
! utils/extmath.py _approximate_mode (BSD-3-Clause). See THIRD_PARTY_NOTICES.md.
module sklearn_resampling
  use sklearn_common, only: dp,require,unique_labels
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: index_split,shuffle_split,stratified_shuffle_split,group_shuffle_split
  public :: repeated_kfold,repeated_stratified_kfold,leave_p_out,leave_p_groups_out
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
end module
