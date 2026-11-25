program reference_classification_report_driver
  use sklearn_common,only:dp
  use sklearn_metrics,only:classification_report,classification_report_result
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  integer::kind,n,q,nlabels,nnames,weighted,digits,dict,i,j,ios
  integer,allocatable::y(:),p(:),ym(:,:),pm(:,:),labels(:)
  character(128),allocatable::names(:)
  real(dp),allocatable::w(:)
  real(dp)::zero
  type(csr_matrix)::a,b
  type(classification_report_result)::r
  do
    read(*,*,iostat=ios)kind,n,q,nlabels,nnames,weighted,digits,dict,zero
    if(ios/=0)exit
    if(kind<=1)then
      allocate(y(n),p(n));read(*,*)y;read(*,*)p
    else
      allocate(ym(n,q),pm(n,q))
      do i=1,n
        read(*,*)ym(i,:);read(*,*)pm(i,:)
      end do
    end if
    if(nlabels>=0)then
      allocate(labels(nlabels));if(nlabels>0)read(*,*)labels
    end if
    if(nnames>=0)then
      allocate(names(nnames))
      do i=1,nnames
        read(*,'(a)')names(i)
      end do
    end if
    if(weighted==1)then
      allocate(w(n));read(*,*)w
    end if
    if(kind>=4)then
      a=csr_from_dense(real(ym,dp));b=csr_from_dense(real(pm,dp))
    end if
    select case(kind)
    case(0);r=classification_report(y,p,labels,names,w,digits,dict==1,zero)
    case(1);r=classification_report(y/=0,p/=0,labels,names,w,digits,dict==1,zero)
    case(2);r=classification_report(ym,pm,labels,names,w,digits,dict==1,zero)
    case(3);r=classification_report(ym/=0,pm/=0,labels,names,w,digits,dict==1,zero)
    case(4);r=classification_report(a,b,labels,names,w,digits,dict==1,zero)
    case(5);r=classification_report(a,pm,labels,names,w,digits,dict==1,zero)
    case(6);r=classification_report(ym,b,labels,names,w,digits,dict==1,zero)
    case(7);r=classification_report(a,pm/=0,labels,names,w,digits,dict==1,zero)
    case(8);r=classification_report(ym/=0,b,labels,names,w,digits,dict==1,zero)
    end select
    write(*,*)size(r%rows),r%n_class_rows,merge(1,0,r%micro_is_accuracy),merge(1,0,r%target_names_mismatch)
    do i=1,size(r%rows)
      write(*,'(a)')r%rows(i)%name
      write(*,'(4(es26.17e3,1x))')r%rows(i)%precision,r%rows(i)%recall,r%rows(i)%f1_score,r%rows(i)%support
    end do
    write(*,'(*(i0,1x))')(iachar(r%text(j:j)),j=1,len(r%text))
    if(allocated(y))deallocate(y,p)
    if(allocated(ym))deallocate(ym,pm)
    if(allocated(labels))deallocate(labels)
    if(allocated(names))deallocate(names)
    if(allocated(w))deallocate(w)
  end do
end program
