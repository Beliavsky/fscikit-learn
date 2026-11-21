! sklearn 1.6.1 metrics/cluster/_supervised.py and
! _expected_mutual_info_fast.pyx contracts, BSD-3-Clause.
! NumPy 2.2.6 pairwise reduction order, BSD-3-Clause.
! Complete upstream notices and research attribution: THIRD_PARTY_NOTICES.md.
module sklearn_cluster_information
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: normalized_mutual_info_score,adjusted_mutual_info_score,fowlkes_mallows_score
  public :: contingency_matrix,pair_confusion_matrix,entropy,expected_mutual_information
  public :: homogeneity_completeness_v_measure
  type :: label_summary
    integer :: n=0
    integer, allocatable :: row(:),column(:)
    integer(int64), allocatable :: row_counts(:),column_counts(:),cells(:)
  end type
  interface contingency_matrix
    module procedure contingency_counts,contingency_smoothed
  end interface
  interface expected_mutual_information
    module procedure expected_default,expected_int64
  end interface
contains
  subroutine heap_sort(values)
    integer(int64), intent(inout) :: values(:)
    integer :: i,last
    integer(int64) :: temp
    do i=size(values)/2,1,-1
      call sift(i,size(values))
    end do
    do last=size(values),2,-1
      temp=values(1); values(1)=values(last); values(last)=temp
      call sift(1,last-1)
    end do
  contains
    subroutine sift(first,last)
      integer, intent(in) :: first,last
      integer :: root,child
      integer(int64) :: temp
      root=first
      do while(root<=last/2)
        child=2*root
        if(child<last) then
          if(values(child)<values(child+1)) child=child+1
        end if
        if(values(root)>=values(child)) exit
        temp=values(root); values(root)=values(child); values(child)=temp; root=child
      end do
    end subroutine
  end subroutine
  subroutine bins(labels,counts,codes)
    integer, intent(in) :: labels(:)
    integer(int64), allocatable, intent(out) :: counts(:)
    integer, allocatable, intent(out) :: codes(:)
    integer(int64), allocatable :: sorted(:),levels(:)
    integer :: i,k,left,right,middle
    call require(size(labels,kind=int64)<huge(i),'cluster information sample count exceeds native index range')
    sorted=int(labels,int64); call heap_sort(sorted)
    allocate(levels(size(labels)),counts(size(labels)),codes(size(labels))); counts=0; k=0
    do i=1,size(labels)
      if(i==1) then
        k=1; levels(k)=sorted(i)
      else if(sorted(i)/=sorted(i-1)) then
        k=k+1; levels(k)=sorted(i)
      end if
      counts(k)=counts(k)+1
    end do
    counts=counts(:k)
    do i=1,size(labels)
      left=1; right=k
      do
        middle=left+(right-left)/2
        if(int(labels(i),int64)==levels(middle)) exit
        if(int(labels(i),int64)<levels(middle)) then
          right=middle-1
        else
          left=middle+1
        end if
      end do
      codes(i)=middle
    end do
  end subroutine
  subroutine summarize(labels_true,labels_pred,info)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    type(label_summary), intent(out) :: info
    integer, allocatable :: a(:),b(:)
    integer(int64), allocatable :: keys(:),distinct(:)
    integer :: i,k,columns
    call require(size(labels_true)==size(labels_pred),'cluster information label count mismatch')
    call bins(labels_true,info%row_counts,a); call bins(labels_pred,info%column_counts,b)
    info%n=size(labels_true); columns=size(info%column_counts)
    allocate(keys(info%n),distinct(info%n),info%cells(info%n)); info%cells=0; k=0
    do i=1,info%n
      keys(i)=int(a(i)-1,int64)*columns+b(i)-1
    end do
    call heap_sort(keys)
    do i=1,info%n
      if(i==1) then
        k=1; distinct(k)=keys(i)
      else if(keys(i)/=keys(i-1)) then
        k=k+1; distinct(k)=keys(i)
      end if
      info%cells(k)=info%cells(k)+1
    end do
    info%cells=info%cells(:k); allocate(info%row(k),info%column(k))
    do i=1,k
      info%row(i)=int(distinct(i)/columns)+1; info%column(i)=int(modulo(distinct(i),int(columns,int64)))+1
    end do
  end subroutine
  recursive real(dp) function pairwise_sum(values) result(total)
    ! NumPy _core/src/umath/loops_utils.h.src, real pairwise reduction.
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,whole,split
    n=size(values)
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=total+values(i)
      end do
    else if(n<=128) then
      partial=values(:8); whole=n-mod(n,8)
      do i=9,whole,8
        partial=partial+values(i:i+7)
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4))) &
        +((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+values(i)
      end do
    else
      split=n/2; split=split-mod(split,8)
      total=pairwise_sum(values(:split))+pairwise_sum(values(split+1:))
    end if
  end function
  real(dp) function count_entropy(counts) result(value)
    integer(int64), intent(in) :: counts(:)
    real(dp) :: n
    if(size(counts)==0) then
      value=1
    else if(size(counts)==1) then
      value=0
    else
      n=real(sum(counts),dp)
      value=-pairwise_sum(real(counts,dp)/n*(log(real(counts,dp))-log(n)))
    end if
  end function
  real(dp) function entropy(labels) result(value)
    integer, intent(in) :: labels(:)
    integer, allocatable :: codes(:)
    integer(int64), allocatable :: counts(:)
    call bins(labels,counts,codes); value=count_entropy(counts)
  end function
  real(dp) function observed_information(info) result(value)
    type(label_summary), intent(in) :: info
    real(dp) :: terms(size(info%cells)),mass,logn
    integer(int64) :: outer
    integer :: i
    value=0
    if(info%n==0.or.size(info%row_counts)==1.or.size(info%column_counts)==1) return
    logn=log(real(info%n,dp))
    do i=1,size(terms)
      mass=real(info%cells(i),dp)/info%n
      outer=info%row_counts(info%row(i))*info%column_counts(info%column(i))
      terms(i)=mass*(log(real(info%cells(i),dp))-logn)+mass*(-log(real(outer,dp))+logn+logn)
      if(abs(terms(i))<epsilon(1._dp)) terms(i)=0
    end do
    value=max(0._dp,pairwise_sum(terms))
  end function
  function contingency_counts(labels_true,labels_pred) result(table)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    integer(int64), allocatable :: table(:,:)
    type(label_summary) :: info
    integer :: i,ierr
    call summarize(labels_true,labels_pred,info)
    allocate(table(size(info%row_counts),size(info%column_counts)),stat=ierr)
    call require(ierr==0,'cluster contingency allocation failed'); table=0
    do i=1,size(info%cells)
      table(info%row(i),info%column(i))=info%cells(i)
    end do
  end function
  function contingency_smoothed(labels_true,labels_pred,eps) result(table)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    real(dp), intent(in) :: eps
    real(dp), allocatable :: table(:,:)
    call require(ieee_is_finite(eps),'cluster contingency eps must be finite')
    call require(eps>=0,'cluster contingency eps must be nonnegative')
    table=real(contingency_counts(labels_true,labels_pred),dp)+eps
  end function
  function pair_confusion_matrix(labels_true,labels_pred) result(table)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    integer(int64) :: table(2,2),squares,n
    type(label_summary) :: info
    call summarize(labels_true,labels_pred,info)
    n=int(info%n,int64); squares=sum(info%cells**2)
    table(2,2)=squares-n
    table(1,2)=sum(info%column_counts**2)-squares
    table(2,1)=sum(info%row_counts**2)-squares
    table(1,1)=n*n-table(1,2)-table(2,1)-squares
  end function
  real(dp) function fowlkes_mallows_score(labels_true,labels_pred) result(value)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    integer(int64) :: pairs(2,2)
    pairs=pair_confusion_matrix(labels_true,labels_pred); value=0
    if(pairs(2,2)>0) value=sqrt(real(pairs(2,2),dp)/real(pairs(2,2)+pairs(1,2),dp)) &
      *sqrt(real(pairs(2,2),dp)/real(pairs(2,2)+pairs(2,1),dp))
  end function
  integer function averaging_kind(average_method) result(kind)
    character(*), intent(in), optional :: average_method
    kind=3
    if(.not.present(average_method)) return
    select case(trim(average_method))
    case('min'); kind=1
    case('geometric'); kind=2
    case('arithmetic'); kind=3
    case('max'); kind=4
    case default; call require(.false.,'unknown cluster MI average_method')
    end select
  end function
  real(dp) function average_entropy(a,b,kind) result(value)
    real(dp), intent(in) :: a,b
    integer, intent(in) :: kind
    select case(kind)
    case(1); value=min(a,b)
    case(2); value=sqrt(a*b)
    case(3); value=(a+b)/2
    case(4); value=max(a,b)
    end select
  end function
  real(dp) function normalized_mutual_info_score(labels_true,labels_pred,average_method) result(value)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    character(*), intent(in), optional :: average_method
    type(label_summary) :: info
    real(dp) :: mi
    integer :: kind
    kind=averaging_kind(average_method); call summarize(labels_true,labels_pred,info)
    value=1
    if(info%n==0.or.(size(info%row_counts)==1.and.size(info%column_counts)==1)) return
    mi=observed_information(info); value=0
    if(mi==0) return
    value=mi/average_entropy(count_entropy(info%row_counts),count_entropy(info%column_counts),kind)
  end function
  real(dp) function expected_counts(a,b,n) result(value)
    integer(int64), intent(in) :: a(:),b(:)
    integer, intent(in) :: n
    real(dp), allocatable :: factorial(:)
    real(dp) :: logn,log_a,log_b,probability,term,g
    integer :: i,j,k,first,last,ai,bj,ierr
    value=0
    if(n==0.or.size(a)==1.or.size(b)==1) return
    call require(n<huge(n),'expected MI sample count exceeds native index range')
    allocate(factorial(0:n),stat=ierr)
    call require(ierr==0,'expected MI workspace allocation failed')
    do i=0,n
      factorial(i)=log_gamma(real(i,dp)+1)
    end do
    logn=log(real(n,dp))
    do i=1,size(a)
      ai=int(a(i)); if(ai==0) cycle
      log_a=log(real(ai,dp))
      do j=1,size(b)
        bj=int(b(j)); if(bj==0) cycle
        log_b=log(real(bj,dp)); first=max(1,ai-n+bj); last=min(ai,bj)
        do k=first,last
          term=(logn+log(real(k,dp)))-log_a-log_b
          g=factorial(ai)+factorial(bj)+factorial(n-ai)+factorial(n-bj) &
            -(factorial(k)+factorial(n))-factorial(ai-k)-factorial(bj-k)-factorial(n-ai-bj+k)
          probability=exp(g); value=value+(real(k,dp)/n)*term*probability
        end do
      end do
    end do
  end function
  real(dp) function expected_int64(contingency,n_samples) result(value)
    integer(int64), intent(in) :: contingency(:,:)
    integer, intent(in) :: n_samples
    integer(int64) :: total
    integer :: i,j
    call require(n_samples>=0,'expected MI n_samples must be nonnegative')
    call require(size(contingency,1)>0.and.size(contingency,2)>0,'expected MI contingency must be nonempty')
    call require(all(contingency>=0),'expected MI counts must be nonnegative')
    total=0
    do j=1,size(contingency,2)
      do i=1,size(contingency,1)
        call require(contingency(i,j)<=int(n_samples,int64)-total,'expected MI count sum mismatch')
        total=total+contingency(i,j)
      end do
    end do
    call require(total==n_samples,'expected MI count sum mismatch')
    value=expected_counts(sum(contingency,dim=2),sum(contingency,dim=1),n_samples)
  end function
  real(dp) function expected_default(contingency,n_samples) result(value)
    integer, intent(in) :: contingency(:,:),n_samples
    value=expected_int64(int(contingency,int64),n_samples)
  end function
  real(dp) function adjusted_mutual_info_score(labels_true,labels_pred,average_method) result(value)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    character(*), intent(in), optional :: average_method
    type(label_summary) :: info
    real(dp) :: mi,expected,denominator
    integer :: kind
    kind=averaging_kind(average_method); call summarize(labels_true,labels_pred,info)
    value=1
    if(info%n==0.or.(size(info%row_counts)==1.and.size(info%column_counts)==1)) return
    mi=observed_information(info); expected=expected_counts(info%row_counts,info%column_counts,info%n)
    denominator=average_entropy(count_entropy(info%row_counts),count_entropy(info%column_counts),kind)-expected
    if(denominator<0) then
      denominator=min(denominator,-epsilon(1._dp))
    else
      denominator=max(denominator,epsilon(1._dp))
    end if
    value=(mi-expected)/denominator
  end function
  subroutine homogeneity_completeness_v_measure(labels_true,labels_pred,homogeneity,completeness,v_measure,beta)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    real(dp), intent(out) :: homogeneity,completeness,v_measure
    real(dp), intent(in), optional :: beta
    type(label_summary) :: info
    real(dp) :: weight,ha,hb,mi,denominator
    weight=1
    if(present(beta)) weight=beta
    call require(ieee_is_finite(weight),'cluster V-measure beta must be finite')
    call require(weight>=0,'cluster V-measure beta must be nonnegative')
    call summarize(labels_true,labels_pred,info)
    homogeneity=1; completeness=1; v_measure=1
    if(info%n==0) return
    ha=count_entropy(info%row_counts); hb=count_entropy(info%column_counts); mi=observed_information(info)
    if(ha/=0) homogeneity=mi/ha
    if(hb/=0) completeness=mi/hb
    v_measure=0
    if(homogeneity+completeness==0) return
    denominator=weight*homogeneity+completeness
    call require(denominator>0,'cluster V-measure beta yields zero denominator')
    v_measure=(1+weight)*homogeneity*completeness/denominator
  end subroutine
end module
