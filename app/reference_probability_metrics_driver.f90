program reference_probability_metrics_driver
  use sklearn_metrics, only: log_loss,d2_log_loss_score
  use sklearn_common, only: dp
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer :: cases,c,op,rep,n,q,np,nc,pred_vector,nl,nw,norm,i,j,pos
  integer, allocatable :: y(:,:),labels(:),original_y(:,:),original_labels(:),ptr(:),cols(:)
  real(dp), allocatable :: p(:,:),weights(:),original_p(:,:),original_w(:),data(:)
  real(dp) :: value
  logical :: normalized,preserved
  type(csr_matrix) :: sparse,saved
  read(*,*) cases
  do c=1,cases
    read(*,*) op,rep,n,q,np,nc,pred_vector,nl,nw,norm
    allocate(y(n,q),p(np,nc));read(*,*) y;read(*,*) p
    if(nl>=0) then
      allocate(labels(nl));read(*,*) labels;original_labels=labels
    end if
    if(nw>=0) then
      allocate(weights(nw));read(*,*) weights;original_w=weights
    end if
    original_y=y;original_p=p;normalized=norm==1
    if(rep>=4) then
      if(rep==4) then
        sparse=csr_from_dense(real(y,dp))
      else
        allocate(ptr(n+1),cols(2*n*q),data(2*n*q));pos=1;ptr(1)=1
        do i=1,n
          do j=q,1,-1
            cols(pos:pos+1)=j;data(pos:pos+1)=real(y(i,j),dp);pos=pos+2
          end do
          ptr(i+1)=pos
        end do
        sparse=csr_from_arrays(n,q,ptr,cols,data);deallocate(ptr,cols,data)
      end if
      saved=sparse
    end if
    write(*,'(A,1X,I0)') 'BEGIN',c
    if(op==0) then
      select case(rep)
      case(0)
        if(pred_vector==1) then
          value=log_loss(y(:,1),p(:,1),labels,weights,normalized)
        else
          value=log_loss(y(:,1),p,labels,weights,normalized)
        end if
      case(1)
        if(pred_vector==1) then
          value=log_loss(y(:,1)/=0,p(:,1),labels,weights,normalized)
        else
          value=log_loss(y(:,1)/=0,p,labels,weights,normalized)
        end if
      case(2)
        if(pred_vector==1) then
          value=log_loss(y,p(:,1),labels,weights,normalized)
        else
          value=log_loss(y,p,labels,weights,normalized)
        end if
      case(3)
        if(pred_vector==1) then
          value=log_loss(y/=0,p(:,1),labels,weights,normalized)
        else
          value=log_loss(y/=0,p,labels,weights,normalized)
        end if
      case(4,5)
        if(pred_vector==1) then
          value=log_loss(sparse,p(:,1),labels,weights,normalized)
        else
          value=log_loss(sparse,p,labels,weights,normalized)
        end if
      end select
    else
      select case(rep)
      case(0)
        if(pred_vector==1) then
          value=d2_log_loss_score(y(:,1),p(:,1),labels,weights)
        else
          value=d2_log_loss_score(y(:,1),p,labels,weights)
        end if
      case(1)
        if(pred_vector==1) then
          value=d2_log_loss_score(y(:,1)/=0,p(:,1),labels,weights)
        else
          value=d2_log_loss_score(y(:,1)/=0,p,labels,weights)
        end if
      case(2)
        if(pred_vector==1) then
          value=d2_log_loss_score(y,p(:,1),labels,weights)
        else
          value=d2_log_loss_score(y,p,labels,weights)
        end if
      case(3)
        if(pred_vector==1) then
          value=d2_log_loss_score(y/=0,p(:,1),labels,weights)
        else
          value=d2_log_loss_score(y/=0,p,labels,weights)
        end if
      case(4,5)
        if(pred_vector==1) then
          value=d2_log_loss_score(sparse,p(:,1),labels,weights)
        else
          value=d2_log_loss_score(sparse,p,labels,weights)
        end if
      end select
    end if
    preserved=all(y==original_y).and.all(transfer(p,[0_int64],size(p))==transfer(original_p,[0_int64],size(p)))
    if(allocated(labels)) preserved=preserved.and.all(labels==original_labels)
    if(allocated(weights)) preserved=preserved.and. &
      all(transfer(weights,[0_int64],size(weights))==transfer(original_w,[0_int64],size(weights)))
    if(rep>=4) preserved=preserved.and.all(sparse%indptr==saved%indptr).and. &
      all(sparse%indices==saved%indices).and.all(sparse%data==saved%data)
    write(*,'(ES26.17E3,1X,I1)') value,merge(1,0,preserved)
    deallocate(y,p)
    if(allocated(labels)) deallocate(labels)
    if(allocated(weights)) deallocate(weights)
  end do
end program
