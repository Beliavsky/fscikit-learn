! Classification margins and discounted ranking adapted from sklearn 1.6.1
! metrics/_classification.py and metrics/_ranking.py (BSD; retained notices).
module sklearn_score_ranking
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total,ieee_mean
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: hinge_loss,top_k_accuracy_score,dcg_score,ndcg_score
  interface hinge_loss
    module procedure hinge_vector,hinge_matrix
  end interface
  interface top_k_accuracy_score
    module procedure top_vector,top_matrix
  end interface
  interface dcg_score
    module procedure dcg_real,dcg_integer,dcg_logical
  end interface
  interface ndcg_score
    module procedure ndcg_real,ndcg_integer,ndcg_logical
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
  function sorted_classes(y) result(classes)
    integer, intent(in) :: y(:)
    integer, allocatable :: classes(:)
    integer :: order(size(y)),work(size(y)),i,n
    call require(size(y)>0,'classification ranking labels must be nonempty')
    order=descending_order(real(y,dp));n=0
    do i=size(y),1,-1
      if(n>0) then
        if(work(n)==y(order(i))) cycle
      end if
      n=n+1;work(n)=y(order(i))
    end do
    classes=work(:n)
  end function
  integer function class_position(classes,label) result(position)
    integer, intent(in) :: classes(:),label
    integer :: left,right,middle
    left=1;right=size(classes);position=0
    do while(left<=right)
      middle=left+(right-left)/2
      if(classes(middle)==label) then
        position=middle;return
      else if(classes(middle)<label) then
        left=middle+1
      else
        right=middle-1
      end if
    end do
  end function
  subroutine check_weights(n,w,normalize)
    integer, intent(in) :: n
    real(dp), optional, intent(in) :: w(:)
    logical, intent(in) :: normalize
    if(.not.present(w)) return
    call require(size(w)==n,'score ranking sample weight length mismatch')
    call require(all(ieee_is_finite(w)),'score ranking sample weights must be finite')
    if(normalize) call require(ieee_total(w)/=0,'score ranking sample weights must have nonzero total')
  end subroutine
  subroutine check_decisions(y,scores,w,normalize)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:,:)
    real(dp), optional, intent(in) :: w(:)
    logical, intent(in) :: normalize
    call require(size(y)>0.and.size(scores,2)>0,'score ranking requires nonempty samples and classes')
    call require(size(scores,1)==size(y),'score ranking target/score sample counts differ')
    call require(all(ieee_is_finite(scores)),'score ranking decisions must be finite')
    call check_weights(size(y),w,normalize)
  end subroutine
  real(dp) function hinge_binary(y,scores,w) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    real(dp), optional, intent(in) :: w(:)
    integer, allocatable :: classes(:)
    real(dp) :: losses(size(y)),sign_,margin
    integer :: i
    classes=sorted_classes(y)
    do i=1,size(y)
      sign_=-1
      if(size(classes)==2) then
        if(y(i)==classes(2)) sign_=1
      else if(size(classes)>2) then
        ! LabelBinarizer's first column in the source's forced-binary branch.
        if(y(i)==classes(1)) sign_=1
      end if
      margin=ieee_product(sign_,scores(i));losses(i)=max(0._dp,ieee_difference(1._dp,margin))
    end do
    value=ieee_mean(losses,w)
  end function
  real(dp) function hinge_vector(y,decision,labels,sample_weight) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: decision(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, allocatable :: classes(:)
    call check_decisions(y,reshape(decision,[size(decision),1]),sample_weight,.true.)
    classes=sorted_classes(y);if(present(labels)) classes=sorted_classes(labels)
    call require(size(classes)<=2,'multiclass hinge decisions require a matrix')
    value=hinge_binary(y,decision,sample_weight)
  end function
  real(dp) function hinge_matrix(y,decision,labels,sample_weight) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: decision(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, allocatable :: classes(:)
    real(dp) :: losses(size(y)),other,margin
    integer :: i,j,index
    call check_decisions(y,decision,sample_weight,.true.)
    classes=sorted_classes(y);if(present(labels)) classes=sorted_classes(labels)
    if(size(classes)<=2) then
      call require(size(decision,2)==1,'binary hinge decisions require one score column')
      value=hinge_binary(y,decision(:,1),sample_weight);return
    end if
    call require(size(classes)==size(decision,2),'hinge score columns must match sorted classes')
    do i=1,size(y)
      index=class_position(classes,y(i));call require(index>0,'hinge target is absent from labels')
      other=-huge(1._dp)
      do j=1,size(classes)
        if(j/=index) other=max(other,decision(i,j))
      end do
      margin=ieee_difference(decision(i,index),other);losses(i)=max(0._dp,ieee_difference(1._dp,margin))
    end do
    value=ieee_mean(losses,sample_weight)
  end function
  function top_classes(y,labels,nclasses) result(classes)
    integer, intent(in) :: y(:),nclasses
    integer, optional, intent(in) :: labels(:)
    integer, allocatable :: classes(:)
    integer :: i
    classes=sorted_classes(y)
    if(present(labels)) then
      classes=sorted_classes(labels)
      call require(size(classes)==size(labels),'top-k labels must be unique')
      call require(all(classes==labels),'top-k labels must be ordered')
    end if
    call require(size(classes)==nclasses,'top-k score columns must match all classes')
    do i=1,size(y)
      call require(class_position(classes,y(i))>0,'top-k target is absent from labels')
    end do
  end function
  real(dp) function hit_average(hits,w,normalize) result(value)
    real(dp), intent(in) :: hits(:)
    real(dp), optional, intent(in) :: w(:)
    logical, intent(in) :: normalize
    real(dp) :: weighted(size(hits))
    integer :: i
    if(normalize) then
      value=ieee_mean(hits,w)
    else if(present(w)) then
      do i=1,size(hits)
        weighted(i)=ieee_product(hits(i),w(i))
      end do
      value=ieee_total(weighted)
    else
      value=ieee_total(hits)
    end if
  end function
  real(dp) function top_vector(y,scores,k,normalize,sample_weight,labels) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    integer, optional, intent(in) :: k,labels(:)
    logical, optional, intent(in) :: normalize
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, allocatable :: classes(:)
    real(dp) :: hits(size(y)),threshold
    integer :: count_,i,predicted
    logical :: norm
    count_=2;norm=.true.;if(present(k)) count_=k
    if(present(normalize)) norm=normalize
    call require(count_>=1,'top-k k must be positive')
    call check_decisions(y,reshape(scores,[size(scores),1]),sample_weight,norm)
    classes=top_classes(y,labels,2);hits=1
    if(count_==1) then
      threshold=0
      if(minval(scores)>=0.and.maxval(scores)<=1) threshold=.5_dp
      do i=1,size(y)
        predicted=1;if(scores(i)>threshold) predicted=2
        hits(i)=merge(1._dp,0._dp,y(i)==classes(predicted))
      end do
    end if
    value=hit_average(hits,sample_weight,norm)
  end function
  real(dp) function top_matrix(y,scores,k,normalize,sample_weight,labels) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:,:)
    integer, optional, intent(in) :: k,labels(:)
    logical, optional, intent(in) :: normalize
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, allocatable :: classes(:)
    real(dp) :: hits(size(y)),own
    integer :: count_,i,j,index,rank
    logical :: norm
    if(size(scores,2)==1) then
      value=top_vector(y,scores(:,1),k,normalize,sample_weight,labels);return
    end if
    count_=2;norm=.true.;if(present(k)) count_=k
    if(present(normalize)) norm=normalize
    call require(count_>=1,'top-k k must be positive')
    call check_decisions(y,scores,sample_weight,norm)
    call require(size(scores,2)>=3,'binary top-k requires one score column')
    classes=top_classes(y,labels,size(scores,2));hits=1
    if(count_<size(classes)) then
      do i=1,size(y)
        index=class_position(classes,y(i));own=scores(i,index);rank=1
        do j=1,size(classes)
          if(scores(i,j)>own) then
            rank=rank+1
          else if(scores(i,j)==own.and.j>index) then
            rank=rank+1
          end if
        end do
        hits(i)=merge(1._dp,0._dp,rank<=count_)
      end do
    end if
    value=hit_average(hits,sample_weight,norm)
  end function
  function discounts(p,k,base,cumulative) result(values)
    integer, intent(in) :: p,k
    real(dp), intent(in) :: base
    logical, intent(in) :: cumulative
    real(dp) :: values(p),log_base
    integer :: j
    log_base=log(base);values=0
    do j=1,min(k,p)
      values(j)=ieee_quotient(1._dp,ieee_quotient(log(real(j,dp)+1),log_base))
    end do
    if(cumulative) then
      do j=2,p
        values(j)=ieee_difference(values(j),-values(j-1))
      end do
    end if
  end function
  real(dp) function row_gain(relevance,scores,discount,ignore_ties) result(value)
    real(dp), intent(in) :: relevance(:),scores(:),discount(:)
    logical, intent(in) :: ignore_ties
    integer :: order(size(scores)),first,last,j,index
    real(dp) :: total,group_discount,previous,term
    order=descending_order(scores);value=0
    if(ignore_ties) then
      ! Break actual ties by higher index, matching reverse stable order.
      first=1
      do while(first<=size(scores))
        last=first
        do while(last<size(scores))
          if(scores(order(last+1))/=scores(order(first))) exit
          last=last+1
        end do
        do j=first,last
          index=order(last-(j-first));term=ieee_product(relevance(index),discount(j))
          value=ieee_difference(value,-term)
        end do
        first=last+1
      end do
      return
    end if
    first=1;previous=0
    do while(first<=size(scores))
      last=first
      do while(last<size(scores))
        if(scores(order(last+1))/=scores(order(first))) exit
        last=last+1
      end do
      total=0
      do j=first,last
        total=ieee_difference(total,-relevance(order(j)))
      end do
      total=ieee_quotient(total,real(last-first+1,dp))
      group_discount=ieee_difference(discount(last),previous);previous=discount(last)
      value=ieee_difference(value,-ieee_product(total,group_discount));first=last+1
    end do
  end function
  real(dp) function gain_metric(y,scores,k,log_base,w,ignore_ties,normalized) result(value)
    real(dp), intent(in) :: y(:,:),scores(:,:)
    integer, optional, intent(in) :: k
    real(dp), intent(in) :: log_base
    real(dp), optional, intent(in) :: w(:)
    logical, optional, intent(in) :: ignore_ties
    logical, intent(in) :: normalized
    real(dp) :: gains(size(y,1)),discount(size(y,2)),ideal_discount(size(y,2)),ideal
    integer :: count_,i
    logical :: ignore
    call require(size(y,1)>0.and.size(y,2)>=2,'DCG requires nonempty samples and at least two documents')
    call require(all(shape(y)==shape(scores)),'DCG relevance/score shapes differ')
    call require(all(ieee_is_finite(y)).and.all(ieee_is_finite(scores)),'DCG relevance and scores must be finite')
    call require(ieee_is_finite(log_base),'DCG logarithm base must be finite')
    call require(log_base>0,'DCG logarithm base must be positive')
    count_=size(y,2);if(present(k)) count_=k
    call require(count_>=1,'DCG k must be positive')
    call check_weights(size(y,1),w,.true.)
    if(normalized) call require(all(y>=0),'NDCG relevance must be nonnegative')
    ignore=.false.;if(present(ignore_ties)) ignore=ignore_ties
    discount=discounts(size(y,2),count_,log_base,.not.ignore)
    if(normalized) ideal_discount=discounts(size(y,2),count_,2._dp,.false.)
    do i=1,size(y,1)
      gains(i)=row_gain(y(i,:),scores(i,:),discount,ignore)
      if(normalized) then
        ideal=row_gain(y(i,:),y(i,:),ideal_discount,.true.)
        if(ideal==0) then
          gains(i)=0
        else
          gains(i)=ieee_quotient(gains(i),ideal)
        end if
      end if
    end do
    value=ieee_mean(gains,w)
  end function
  real(dp) function dcg_real(y,scores,k,log_base,sample_weight,ignore_ties) result(value)
    real(dp), intent(in) :: y(:,:),scores(:,:)
    integer, optional, intent(in) :: k
    real(dp), optional, intent(in) :: log_base,sample_weight(:)
    logical, optional, intent(in) :: ignore_ties
    real(dp) :: base
    base=2;if(present(log_base)) base=log_base
    value=gain_metric(y,scores,k,base,sample_weight,ignore_ties,.false.)
  end function
  real(dp) function ndcg_real(y,scores,k,sample_weight,ignore_ties) result(value)
    real(dp), intent(in) :: y(:,:),scores(:,:)
    integer, optional, intent(in) :: k
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: ignore_ties
    value=gain_metric(y,scores,k,2._dp,sample_weight,ignore_ties,.true.)
  end function
  real(dp) function dcg_integer(y,scores,k,log_base,sample_weight,ignore_ties) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    integer, optional, intent(in) :: k
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: ignore_ties
    real(dp), optional, intent(in) :: log_base
    value=dcg_real(real(y,dp),scores,k,log_base,sample_weight,ignore_ties)
  end function
  real(dp) function dcg_logical(y,scores,k,log_base,sample_weight,ignore_ties) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    integer, optional, intent(in) :: k
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: ignore_ties
    real(dp), optional, intent(in) :: log_base
    value=dcg_real(real(merge(1,0,y),dp),scores,k,log_base,sample_weight,ignore_ties)
  end function
  real(dp) function ndcg_integer(y,scores,k,sample_weight,ignore_ties) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    integer, optional, intent(in) :: k
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: ignore_ties
    value=ndcg_real(real(y,dp),scores,k,sample_weight,ignore_ties)
  end function
  real(dp) function ndcg_logical(y,scores,k,sample_weight,ignore_ties) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: scores(:,:)
    integer, optional, intent(in) :: k
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: ignore_ties
    value=ndcg_real(real(merge(1,0,y),dp),scores,k,sample_weight,ignore_ties)
  end function
end module
