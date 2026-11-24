! Multilabel ranking measures adapted from sklearn 1.6.1 metrics/_ranking.py (BSD).
! Native grouped merge sorting, interfaces and CSR guards are MIT contributions.
module sklearn_multilabel_ranking
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix
  use sklearn_ieee_math, only: ieee_mean,ieee_total
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: coverage_error,label_ranking_average_precision_score,label_ranking_loss
  interface coverage_error
    module procedure coverage_integer,coverage_logical
  end interface
  interface label_ranking_average_precision_score
    module procedure precision_integer,precision_logical,precision_csr
  end interface
  interface label_ranking_loss
    module procedure loss_integer,loss_logical,loss_csr
  end interface
contains
  function descending_order(scores) result(order)
    real(dp), intent(in) :: scores(:)
    integer :: order(size(scores)),work(size(scores)),n,width,left,middle,right,i,j,k
    n=size(scores);order=[(i,i=1,n)];width=1
    do while(width<n)
      left=1
      do while(left<=n)
        middle=left+min(width,n-left+1);right=middle+min(width,n-middle+1)
        i=left;j=middle
        do k=left,right-1
          if(i>=middle) then
            work(k)=order(j);j=j+1
          else if(j>=right) then
            work(k)=order(i);i=i+1
          else if(scores(order(i))>=scores(order(j))) then
            work(k)=order(i);i=i+1
          else
            work(k)=order(j);j=j+1
          end if
        end do
        left=right
      end do
      order=work
      if(width>=n-width) exit
      width=width*2
    end do
  end function
  real(dp) function row_score(relevant,scores,kind) result(value)
    logical, intent(in) :: relevant(:)
    real(dp), intent(in) :: scores(:)
    integer, intent(in) :: kind
    integer :: order(size(scores)),first,last,positive,group_positive,npositive,nlabels,i
    nlabels=size(scores);npositive=count(relevant);value=0
    if(npositive==0.or.npositive==nlabels) then
      if(kind==1) value=real(npositive,dp)
      if(kind==2) value=1
      return
    end if
    order=descending_order(scores);first=1;positive=0
    do while(first<=nlabels)
      last=first
      do while(last<nlabels)
        if(scores(order(last+1))/=scores(order(first))) exit
        last=last+1
      end do
      group_positive=0
      do i=first,last
        if(relevant(order(i))) group_positive=group_positive+1
      end do
      positive=positive+group_positive
      select case(kind)
      case(1)
        if(group_positive>0) value=real(last,dp)
      case(2)
        value=value+real(group_positive,dp)*positive/last
      case(3)
        ! Every relevant/irrelevant tied pair is an error, as in sklearn.
        value=value+real(group_positive,dp)*(last-positive)
      end select
      first=last+1
    end do
    if(kind==2) value=value/npositive
    if(kind==3) value=value/(real(npositive,dp)*(nlabels-npositive))
  end function
  subroutine validate(n,p,scores,kind,sample_weight)
    integer, intent(in) :: n,p,kind
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call require(n>0.and.p>0,'multilabel ranking requires nonempty samples and labels')
    call require(kind==2.or.p>=2,'multilabel ranking requires at least two labels')
    call require(all(shape(scores)==[n,p]),'multilabel ranking target/score shapes differ')
    call require(all(ieee_is_finite(scores)),'multilabel ranking scores must be finite')
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'multilabel ranking sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'multilabel ranking weights must be finite')
      call require(ieee_total(sample_weight)/=0,'multilabel ranking weights must have nonzero total')
    end if
  end subroutine
  real(dp) function dense_score(y,scores,kind,sample_weight) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    integer, intent(in) :: kind
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: values(size(y,1))
    integer :: i
    call validate(size(y,1),size(y,2),scores,kind,sample_weight)
    do i=1,size(y,1)
      values(i)=row_score(y(i,:),scores(i,:),kind)
    end do
    value=ieee_mean(values,sample_weight)
  end function
  real(dp) function integer_score(y,scores,kind,sample_weight) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    integer, intent(in) :: kind
    real(dp), optional, intent(in) :: sample_weight(:)
    integer :: first,second,i,j
    logical :: has_second
    call validate(size(y,1),size(y,2),scores,kind,sample_weight)
    first=y(1,1);second=first;has_second=.false.
    do j=1,size(y,2)
      do i=1,size(y,1)
        if(y(i,j)==first) cycle
        if(.not.has_second) then
          second=y(i,j);has_second=.true.
        end if
        call require(y(i,j)==second,'multilabel ranking targets must be an indicator with at most two values')
      end do
    end do
    value=dense_score(y/=0,scores,kind,sample_weight)
  end function
  real(dp) function sparse_score(y,scores,kind,sample_weight) result(value)
    type(csr_matrix), intent(in) :: y
    real(dp), intent(in) :: scores(:,:)
    integer, intent(in) :: kind
    real(dp), optional, intent(in) :: sample_weight(:)
    logical :: relevant(y%n_cols)
    real(dp) :: values(y%n_rows)
    integer :: i,start,stop
    call y%validate();call validate(y%n_rows,y%n_cols,scores,kind,sample_weight)
    call require(all(y%data==1._dp),'multilabel ranking CSR indicators must store ones')
    do i=1,y%n_rows
      start=y%indptr(i);stop=y%indptr(i+1)-1
      if(stop>start) call require(all(y%indices(start+1:stop)>y%indices(start:stop-1)), &
        'multilabel ranking CSR indices must be sorted and unique')
      relevant=.false.
      if(stop>=start) relevant(y%indices(start:stop))=.true.
      values(i)=row_score(relevant,scores(i,:),kind)
    end do
    value=ieee_mean(values,sample_weight)
  end function
  real(dp) function coverage_integer(y,scores,sample_weight) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=integer_score(y,scores,1,sample_weight)
  end function
  real(dp) function coverage_logical(y,scores,sample_weight) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=dense_score(y,scores,1,sample_weight)
  end function
  real(dp) function precision_integer(y,scores,sample_weight) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=integer_score(y,scores,2,sample_weight)
  end function
  real(dp) function precision_logical(y,scores,sample_weight) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=dense_score(y,scores,2,sample_weight)
  end function
  real(dp) function precision_csr(y,scores,sample_weight) result(value)
    type(csr_matrix), intent(in) :: y
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=sparse_score(y,scores,2,sample_weight)
  end function
  real(dp) function loss_integer(y,scores,sample_weight) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=integer_score(y,scores,3,sample_weight)
  end function
  real(dp) function loss_logical(y,scores,sample_weight) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=dense_score(y,scores,3,sample_weight)
  end function
  real(dp) function loss_csr(y,scores,sample_weight) result(value)
    type(csr_matrix), intent(in) :: y
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=sparse_score(y,scores,3,sample_weight)
  end function
end module
